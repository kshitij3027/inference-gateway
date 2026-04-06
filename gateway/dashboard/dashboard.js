(function () {
    "use strict";

    // --- Color palette for backends ---
    var COLORS = [
        "#4caf50", "#2196f3", "#ff9800", "#e91e63", "#9c27b0",
        "#00bcd4", "#ff5722", "#8bc34a", "#3f51b5", "#cddc39",
        "#f44336", "#009688", "#673ab7", "#ffc107"
    ];

    // --- State ---
    var state = {
        ws: null,
        reconnectDelay: 1000,
        backends: {},
        tenantRequests: {},
        cacheHits: 0,
        cacheMisses: 0,
        activeRequests: 0,
        completedRequests: 0,
        rateLimitedRequests: 0,
        tenantChart: null,
        cacheChart: null,
        backendColors: {},
        colorIndex: 0
    };

    function getBackendColor(name) {
        if (!state.backendColors[name]) {
            state.backendColors[name] = COLORS[state.colorIndex % COLORS.length];
            state.colorIndex++;
        }
        return state.backendColors[name];
    }

    // --- WebSocket ---
    function connect() {
        var protocol = location.protocol === "https:" ? "wss:" : "ws:";
        var wsUrl = protocol + "//" + location.host + "/ws/dashboard";
        state.ws = new WebSocket(wsUrl);
        setStatus("connecting");

        state.ws.onopen = function () {
            setStatus("connected");
            state.reconnectDelay = 1000;
            fetchInitialState();
        };

        state.ws.onmessage = function (event) {
            var msg = JSON.parse(event.data);
            handleEvent(msg.type, msg.data);
        };

        state.ws.onclose = function () {
            setStatus("disconnected");
            setTimeout(function () {
                state.reconnectDelay = Math.min(state.reconnectDelay * 2, 30000);
                connect();
            }, state.reconnectDelay);
        };

        state.ws.onerror = function () {
            state.ws.close();
        };
    }

    function setStatus(status) {
        var dot = document.getElementById("status-dot");
        var text = document.getElementById("status-text");
        dot.className = status;
        var labels = { connected: "Connected", disconnected: "Disconnected", connecting: "Connecting..." };
        text.textContent = labels[status] || status;
    }

    // --- Event handling ---
    function handleEvent(type, data) {
        switch (type) {
            case "new_request": onNewRequest(data); break;
            case "request_complete": onRequestComplete(data); break;
            case "cache_hit": onCacheHit(data); break;
            case "cache_miss": onCacheMiss(data); break;
            case "circuit_state_change": onCircuitStateChange(data); break;
            case "rate_limit_hit": onRateLimitHit(data); break;
        }
    }

    // --- Backend Health ---
    function onCircuitStateChange(data) {
        if (state.backends[data.backend]) {
            state.backends[data.backend].state = data.new_state;
        } else {
            state.backends[data.backend] = { state: data.new_state, error_rate: 0 };
        }
        renderBackendHealth();
    }

    function renderBackendHealth() {
        var grid = document.getElementById("health-grid");
        grid.innerHTML = "";
        var names = Object.keys(state.backends).sort();
        for (var i = 0; i < names.length; i++) {
            var name = names[i];
            var info = state.backends[name];
            var card = document.createElement("div");
            card.className = "backend-card";
            card.innerHTML =
                '<div class="health-dot ' + (info.state || "CLOSED") + '"></div>' +
                '<div class="backend-name" title="' + name + '">' + name.replace(/^mock-/, "") + '</div>' +
                '<div class="backend-error-rate">' + ((info.error_rate || 0) * 100).toFixed(0) + '% err</div>';
            grid.appendChild(card);
        }
    }

    // --- Hash Ring ---
    function renderHashRing(ringData) {
        var svg = document.getElementById("ring-svg");
        var legend = document.getElementById("ring-legend");
        svg.innerHTML = "";
        legend.innerHTML = "";

        // Pick first model's ring data
        var models = Object.keys(ringData);
        if (models.length === 0) return;

        var modelName = models[0];
        var dist = ringData[modelName].distribution || {};
        var total = ringData[modelName].total_vnodes || 0;
        if (total === 0) return;

        var cx = 150, cy = 150, r = 120;
        var startAngle = -Math.PI / 2;
        var backends = Object.keys(dist).sort();

        for (var i = 0; i < backends.length; i++) {
            var bName = backends[i];
            var count = dist[bName];
            var sweep = (count / total) * 2 * Math.PI;
            var endAngle = startAngle + sweep;
            var color = getBackendColor(bName);

            var x1 = cx + r * Math.cos(startAngle);
            var y1 = cy + r * Math.sin(startAngle);
            var x2 = cx + r * Math.cos(endAngle);
            var y2 = cy + r * Math.sin(endAngle);
            var largeArc = sweep > Math.PI ? 1 : 0;

            var path = document.createElementNS("http://www.w3.org/2000/svg", "path");
            path.setAttribute("d",
                "M " + cx + " " + cy +
                " L " + x1.toFixed(2) + " " + y1.toFixed(2) +
                " A " + r + " " + r + " 0 " + largeArc + " 1 " + x2.toFixed(2) + " " + y2.toFixed(2) +
                " Z"
            );
            path.setAttribute("fill", color);
            path.setAttribute("stroke", "#0a0a1a");
            path.setAttribute("stroke-width", "1");
            svg.appendChild(path);

            var item = document.createElement("div");
            item.className = "legend-item";
            item.innerHTML =
                '<span class="legend-color" style="background:' + color + '"></span>' +
                '<span>' + bName.replace(/^mock-/, "") + ' (' + count + ')</span>';
            legend.appendChild(item);

            startAngle = endAngle;
        }

        // Add model label in center
        var text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", cx);
        text.setAttribute("y", cy);
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("dominant-baseline", "middle");
        text.setAttribute("fill", "#aaa");
        text.setAttribute("font-size", "11");
        text.textContent = modelName;
        svg.appendChild(text);
    }

    // --- Request Flow ---
    function onNewRequest(data) {
        state.activeRequests++;
        updateFlowStats();
        highlightStage("auth");

        // Update tenant usage
        state.tenantRequests[data.tenant_id] = (state.tenantRequests[data.tenant_id] || 0) + 1;
        updateTenantChart();
    }

    function onRequestComplete(data) {
        state.activeRequests = Math.max(0, state.activeRequests - 1);
        state.completedRequests++;
        updateFlowStats();
        highlightStage("response");
    }

    function onRateLimitHit(data) {
        state.rateLimitedRequests++;
        updateFlowStats();
        highlightStage("rate-limit");
    }

    function highlightStage(stageName) {
        var stage = document.querySelector('.flow-stage[data-stage="' + stageName + '"]');
        if (!stage) return;
        stage.classList.add("active");
        setTimeout(function () { stage.classList.remove("active"); }, 500);
    }

    function updateFlowStats() {
        document.getElementById("active-count").textContent = state.activeRequests;
        document.getElementById("completed-count").textContent = state.completedRequests;
        document.getElementById("ratelimited-count").textContent = state.rateLimitedRequests;
    }

    // --- Tenant Usage Chart ---
    function updateTenantChart() {
        var labels = Object.keys(state.tenantRequests).sort();
        var data = labels.map(function (t) { return state.tenantRequests[t]; });
        var colors = labels.map(function (_, i) { return COLORS[i % COLORS.length]; });

        if (!state.tenantChart) {
            var ctx = document.getElementById("tenant-chart").getContext("2d");
            state.tenantChart = new Chart(ctx, {
                type: "bar",
                data: {
                    labels: labels,
                    datasets: [{
                        label: "Requests",
                        data: data,
                        backgroundColor: colors,
                        borderWidth: 0
                    }]
                },
                options: {
                    indexAxis: "y",
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { ticks: { color: "#aaa" }, grid: { color: "#2a2a4a" } },
                        y: { ticks: { color: "#aaa" }, grid: { display: false } }
                    }
                }
            });
        } else {
            state.tenantChart.data.labels = labels;
            state.tenantChart.data.datasets[0].data = data;
            state.tenantChart.data.datasets[0].backgroundColor = colors;
            state.tenantChart.update();
        }
    }

    // --- Cache Gauge ---
    function onCacheHit(data) {
        state.cacheHits++;
        updateCacheGauge();
    }

    function onCacheMiss(data) {
        state.cacheMisses++;
        updateCacheGauge();
    }

    function updateCacheGauge() {
        var total = state.cacheHits + state.cacheMisses;
        var hitRate = total > 0 ? ((state.cacheHits / total) * 100).toFixed(1) : "0.0";

        if (!state.cacheChart) {
            var ctx = document.getElementById("cache-gauge").getContext("2d");
            state.cacheChart = new Chart(ctx, {
                type: "doughnut",
                data: {
                    labels: ["Hits", "Misses"],
                    datasets: [{
                        data: [state.cacheHits, state.cacheMisses],
                        backgroundColor: ["#4caf50", "#f44336"],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: "70%",
                    plugins: {
                        legend: { position: "bottom", labels: { color: "#aaa", padding: 12 } }
                    }
                }
            });
        } else {
            state.cacheChart.data.datasets[0].data = [state.cacheHits, state.cacheMisses];
            state.cacheChart.update();
        }

        document.getElementById("cache-stats-text").textContent =
            hitRate + "% hit rate (" + state.cacheHits + " hits / " + state.cacheMisses + " misses)";
    }

    // --- Initial state fetch ---
    function fetchInitialState() {
        fetch("/admin/backends")
            .then(function (r) { return r.json(); })
            .then(function (data) {
                for (var i = 0; i < data.length; i++) {
                    var b = data[i];
                    state.backends[b.name] = {
                        state: b.health || b.circuit_breaker.state,
                        error_rate: b.circuit_breaker.error_rate
                    };
                }
                renderBackendHealth();
            })
            .catch(function () {});

        fetch("/admin/ring")
            .then(function (r) { return r.json(); })
            .then(function (data) { renderHashRing(data); })
            .catch(function () {});

        fetch("/admin/cache/stats")
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.enabled) {
                    state.cacheHits = data.hits || 0;
                    state.cacheMisses = data.misses || 0;
                    updateCacheGauge();
                }
            })
            .catch(function () {});
    }

    // --- Bootstrap ---
    connect();
})();
