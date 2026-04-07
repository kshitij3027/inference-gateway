"""Inference Gateway admin CLI tool."""

import sys

import click
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _get(ctx, path: str):
    """GET request to the gateway. Returns parsed JSON or exits on error."""
    url = ctx.obj["gateway"] + path
    try:
        resp = httpx.get(url, timeout=10.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Cannot connect to gateway at {ctx.obj['gateway']}")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error:[/red] {e.response.status_code} from {url}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def _delete(ctx, path: str):
    """DELETE request to the gateway."""
    url = ctx.obj["gateway"] + path
    try:
        resp = httpx.delete(url, timeout=10.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Cannot connect to gateway at {ctx.obj['gateway']}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@click.group()
@click.option("--gateway", "-g", default="http://localhost:8080", envvar="IGW_GATEWAY_URL",
              help="Gateway base URL")
@click.pass_context
def cli(ctx, gateway):
    """Inference Gateway admin CLI."""
    ctx.ensure_object(dict)
    ctx.obj["gateway"] = gateway.rstrip("/")


@cli.command()
@click.pass_context
def status(ctx):
    """Show gateway health and readiness."""
    health = _get(ctx, "/health")
    ready = _get(ctx, "/ready")

    status_str = f"Health: [green]{health.get('status', 'unknown')}[/green]"
    backends_str = f"Backends: {ready.get('healthy_backends', '?')}/{ready.get('total_backends', '?')} healthy"

    console.print(Panel(f"{status_str}\n{backends_str}", title="Gateway Status"))


@cli.command()
@click.pass_context
def backends(ctx):
    """List all backends with health status."""
    data = _get(ctx, "/admin/backends")

    table = Table(title="Backends")
    table.add_column("Name", style="cyan")
    table.add_column("Provider")
    table.add_column("Models")
    table.add_column("Health")
    table.add_column("Error Rate")
    table.add_column("Requests")

    for b in data:
        cb = b.get("circuit_breaker", {})
        state = b.get("health", cb.get("state", "?"))
        color = {"CLOSED": "green", "OPEN": "red", "HALF_OPEN": "yellow"}.get(state, "white")
        table.add_row(
            b["name"],
            b["provider"],
            ", ".join(b.get("models", [])),
            f"[{color}]{state}[/{color}]",
            f"{cb.get('error_rate', 0):.0%}",
            str(cb.get("requests_in_window", 0)),
        )

    console.print(table)


@cli.command()
@click.pass_context
def tenants(ctx):
    """List configured tenants."""
    data = _get(ctx, "/admin/tenants")

    table = Table(title="Tenants")
    table.add_column("ID", style="cyan")
    table.add_column("Models")
    table.add_column("Priority")
    table.add_column("RPS Limit")
    table.add_column("RPM Limit")
    table.add_column("Daily Token Budget")

    for t in data:
        models = t.get("allowed_models", [])
        models_str = ", ".join(models) if models != ["*"] else "[dim]*[/dim]"
        table.add_row(
            t["id"],
            models_str,
            str(t.get("priority", 1)),
            str(t.get("rate_limit_rps") or "-"),
            str(t.get("rate_limit_rpm") or "-"),
            str(t.get("token_budget_daily") or "-"),
        )

    console.print(table)


@cli.command()
@click.pass_context
def ring(ctx):
    """Show consistent hash ring state."""
    data = _get(ctx, "/admin/ring")

    for model, info in data.items():
        table = Table(title=f"Hash Ring: {model}")
        table.add_column("Backend", style="cyan")
        table.add_column("VNodes", justify="right")
        table.add_column("Share", justify="right")

        dist = info.get("distribution", {})
        total = info.get("total_vnodes", 1)
        for backend, vnodes in sorted(dist.items()):
            share = (vnodes / total * 100) if total > 0 else 0
            table.add_row(backend, str(vnodes), f"{share:.1f}%")

        console.print(table)


@cli.command()
@click.pass_context
def journal(ctx):
    """Show recent journal entries."""
    data = _get(ctx, "/admin/journal")

    if not data.get("enabled"):
        console.print("[yellow]Journal is disabled (requires Redis)[/yellow]")
        return

    entries = data.get("entries", [])
    table = Table(title=f"Journal ({data.get('count', len(entries))} entries)")
    table.add_column("Time")
    table.add_column("Request ID", style="dim")
    table.add_column("Tenant", style="cyan")
    table.add_column("Model")
    table.add_column("Backend")
    table.add_column("Status", justify="right")
    table.add_column("Latency", justify="right")

    for e in entries:
        status_val = str(e.get("status", ""))
        color = "green" if status_val == "200" else "red" if status_val.startswith(("4", "5")) else "white"
        latency = e.get("latency_ms")
        try:
            latency_str = f"{float(latency):.0f}ms" if latency is not None else "-"
        except (ValueError, TypeError):
            latency_str = str(latency)
        ts = e.get("timestamp", "")
        try:
            from datetime import datetime, timezone
            ts_str = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%H:%M:%S") if ts else "-"
        except (ValueError, TypeError):
            ts_str = str(ts)[:19] if ts else "-"
        table.add_row(
            ts_str,
            (str(e.get("request_id", ""))[:12] + "...") if e.get("request_id") else "-",
            e.get("tenant_id", "-"),
            e.get("model", "-"),
            e.get("backend", "-"),
            f"[{color}]{status_val}[/{color}]",
            latency_str,
        )

    console.print(table)


@cli.group()
def cache():
    """Cache operations."""
    pass


@cache.command("stats")
@click.pass_context
def cache_stats(ctx):
    """Show cache statistics."""
    data = _get(ctx, "/admin/cache/stats")

    if not data.get("enabled"):
        console.print("[yellow]Cache is disabled (requires Redis)[/yellow]")
        return

    hits = data.get("hits", 0)
    misses = data.get("misses", 0)
    total = hits + misses
    rate = (hits / total * 100) if total > 0 else 0

    lines = [
        f"Hit Rate:  [green]{rate:.1f}%[/green]",
        f"Hits:      {hits}",
        f"Misses:    {misses}",
        f"Entries:   {data.get('entries', 0)}",
    ]
    l1 = data.get("l1_stats") or data.get("l1_hits") is not None
    if l1:
        l1_hits = data.get("l1_hits", data.get("l1_stats", {}).get("hits", 0))
        l1_misses = data.get("l1_misses", data.get("l1_stats", {}).get("misses", 0))
        lines.append(f"L1 Hits:   {l1_hits}")
        lines.append(f"L1 Misses: {l1_misses}")

    console.print(Panel("\n".join(lines), title="Cache Stats"))


@cache.command("flush")
@click.pass_context
def cache_flush(ctx):
    """Flush all cached responses."""
    if not click.confirm("Are you sure you want to flush the cache?"):
        console.print("Cancelled.")
        return
    data = _delete(ctx, "/admin/cache")
    console.print(f"[green]Cache flushed.[/green] Entries deleted: {data.get('entries_deleted', '?')}")


@cli.command()
@click.option("--tenant", "-t", help="Filter by tenant ID")
@click.option("--days", "-d", default=7, help="Number of days to show")
@click.pass_context
def cost(ctx, tenant, days):
    """Show estimated cost summary."""
    params = f"?days={days}"
    if tenant:
        params += f"&tenant={tenant}"
    data = _get(ctx, f"/admin/cost{params}")

    if not data.get("enabled"):
        console.print("[yellow]Cost tracking is disabled (requires Redis)[/yellow]")
        return

    if tenant:
        # Single tenant view
        table = Table(title=f"Cost: {data.get('tenant_id', tenant)}")
        table.add_column("Date")
        table.add_column("Cost ($)", justify="right")
        costs = data.get("costs_by_date", {})
        for date_str in sorted(costs.keys(), reverse=True):
            val = costs[date_str]
            table.add_row(date_str, f"${val:.6f}")
        console.print(table)
        console.print(f"Today: [green]${data.get('today', 0):.6f}[/green]")
    else:
        # All tenants summary
        tenants_data = data.get("tenants", [])
        table = Table(title="Cost Summary (All Tenants)")
        table.add_column("Tenant", style="cyan")
        table.add_column("Today ($)", justify="right")
        table.add_column(f"{days}d Total ($)", justify="right")

        for t in tenants_data:
            today_cost = t.get("today", 0)
            total = sum(t.get("costs_by_date", {}).values())
            table.add_row(
                t["tenant_id"],
                f"${today_cost:.6f}",
                f"${total:.6f}",
            )

        console.print(table)
