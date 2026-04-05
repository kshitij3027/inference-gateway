# Production Deployment Guide

This document covers deploying the Inference Gateway to AWS using the Terraform definitions in `terraform/`. It maps each local Docker Compose component to its production AWS equivalent, provides cost estimates at different traffic levels, and details scaling, caching, and monitoring strategies.

---

## Prerequisites

- **AWS account** with permissions for ECS, ElastiCache, ALB, VPC, CloudWatch, ECR, SSM, and IAM
- **Terraform >= 1.5** installed locally
- **Docker** installed (for building and pushing images)
- **ACM certificate** (optional, required for HTTPS on the ALB)
- **Domain name** (optional, for Route53 DNS routing to the ALB)

---

## Deployment Steps (High-Level)

1. **Build and push** the Docker image to ECR
2. **Configure** `terraform.tfvars` from the provided example
3. Run `terraform init`, `terraform plan`, and `terraform apply`
4. **Verify** ALB health checks pass and targets are healthy
5. **Point DNS** to the ALB (optional, via Route53 or external DNS)

---

## Local vs Production Comparison

| Component | Local (Docker Compose) | Production (AWS) | Notes |
|---|---|---|---|
| Gateway Instances | 3x containers, direct build | ECS Fargate ARM64 (Graviton) | 20% cost savings with Graviton |
| Load Balancer | Nginx reverse proxy (port 8080) | ALB with HTTPS/TLS 1.3 | SSL termination at ALB |
| Redis | Single Redis 7 Alpine container | ElastiCache Redis 7.1 replication group | Multi-AZ, encryption at rest + in transit |
| Monitoring | Prometheus + Grafana (self-hosted) | CloudWatch logs + alarms | Optional: add self-hosted Prometheus |
| Networking | Docker bridge network | VPC with public/private subnets (2 AZs) | NAT Gateway for outbound from private subnets |
| Secrets | .env files, docker-compose env vars | SSM Parameter Store | Fetched at task startup via execution role |
| Scaling | Manual (edit docker-compose replicas) | ECS auto-scaling (CPU target tracking) | Min 2, max 10 tasks |
| TLS/SSL | None | ACM certificate on ALB | Free with ACM |
| Health Checks | Nginx passive (max_fails=3) | ALB active probes on /health every 30s | Automatic target deregistration |
| Image Registry | Local Docker build | ECR with vulnerability scanning | Lifecycle policy cleans old images |
| Log Management | stdout to Docker logs | CloudWatch Logs (30-day retention) | Structured JSON logs |
| Service Discovery | Docker Compose service names | ALB DNS name | Optional Route53 for custom domain |

---

## Cost Estimates

> All estimates use **US East 1 (N. Virginia)** pricing as of 2025. Actual costs vary by region and usage patterns. Graviton (ARM64) instances are used where available for cost optimization.

### At 100 RPS (Low-Medium Traffic)

| Service | Configuration | Monthly Cost |
|---|---|---|
| ECS Fargate | 3x tasks, 1 vCPU / 2 GB (ARM64) | ~$91 |
| ElastiCache Redis | 1x cache.t4g.micro (0.5 GB) | ~$12 |
| ALB | 1 ALB + ~3 LCU average | ~$34 |
| NAT Gateway | 1x (single) + ~50 GB data transfer | ~$35 |
| CloudWatch | Logs (~5 GB) + 7 alarms | ~$8 |
| ECR | ~2 GB stored | ~$0.20 |
| **Total** | | **~$180/month** |

**Cost optimization tips for 100 RPS:**

- Use Fargate Spot for non-critical workloads (60-70% savings)
- Single NAT Gateway is sufficient (vs per-AZ)
- `cache.t4g.micro` is adequate for light caching load
- VPC endpoints reduce NAT data transfer costs

### At 1000 RPS (Medium-High Traffic)

| Service | Configuration | Monthly Cost |
|---|---|---|
| ECS Fargate | 8x tasks, 2 vCPU / 4 GB (ARM64) | ~$487 |
| ElastiCache Redis | 2x cache.r7g.large (13 GB, Primary + Replica) | ~$260 |
| ALB | 1 ALB + ~15 LCU average | ~$104 |
| NAT Gateway | 2x (one per AZ) + ~300 GB data transfer | ~$80 |
| CloudWatch | Logs (~30 GB) + 7 alarms + dashboards | ~$25 |
| ECR | ~3 GB stored | ~$0.30 |
| VPC Endpoints | 3x Interface endpoints | ~$22 |
| **Total** | | **~$978/month** |

**Cost optimization tips for 1000 RPS:**

- Reserved ElastiCache nodes: 1-year = ~30% savings, 3-year = ~50%
- Fargate Spot for burst capacity (keep on-demand for baseline)
- VPC endpoints save significant NAT transfer costs at this scale
- Consider Savings Plans for predictable Fargate usage

---

## Redis Cluster Guidance

### When to Use Single Node vs Replication Group

| Factor | Single Node (Dev/Test) | Replication Group (Production) |
|---|---|---|
| Use case | Development, testing, low traffic | Production, any traffic level |
| Failover | Manual intervention, downtime | Automatic, < 30 seconds |
| Read scaling | None | Read replicas for read-heavy workloads |
| Data durability | Snapshots only | Real-time replication across AZs |
| Cost | 1x node price | 2x+ node price |
| Terraform var | `redis_num_cache_clusters = 1` | `redis_num_cache_clusters = 2` (default) |

### Eviction Policy

The gateway uses `allkeys-lru` (Least Recently Used) eviction. This is correct because:

- Semantic cache entries are re-computable (cache miss = slower but not broken)
- Rate limiter keys have TTLs and expire naturally
- LRU keeps the most relevant cached responses in memory

### When to Scale Redis

Monitor these CloudWatch metrics:

- **EngineCPUUtilization > 65%**: Redis is single-threaded; high CPU means saturation. Scale up node type.
- **DatabaseMemoryUsagePercentage > 80%**: Approaching eviction pressure. Either scale up node type or reduce cache TTLs.
- **CurrConnections**: If approaching `maxclients` (default 65000), you may have a connection leak.
- **CacheHitRate < 50%**: Consider increasing memory to hold more cached responses.

### Sizing Guide

| Traffic Level | Node Type | Memory | Approximate Entries | Monthly Cost |
|---|---|---|---|---|
| < 100 RPS | cache.t4g.micro | 0.5 GB | ~5,000 cached responses | $12 |
| 100-500 RPS | cache.t4g.medium | 3.1 GB | ~30,000 cached responses | $50 |
| 500-2000 RPS | cache.r7g.large | 13 GB | ~130,000 cached responses | $130 |
| 2000+ RPS | cache.r7g.xlarge | 26 GB | ~260,000 cached responses | $260 |

### Connection Pooling

The gateway uses async Redis connections via `redis-py`. In production with ElastiCache:

- Enable transit encryption (TLS) -- the Terraform config enables this by default
- Connection string format changes to `rediss://` (double 's' for TLS)
- The ECS task definition automatically constructs the correct `REDIS_URL`

---

## Scaling Strategy

### Gateway (ECS) Scaling

**Auto-scaling configuration:**

- Metric: ECS average CPU utilization
- Target: 70%
- Min tasks: 2 (high availability)
- Max tasks: 10
- Scale-out cooldown: 60 seconds (react quickly to traffic spikes)
- Scale-in cooldown: 300 seconds (avoid flapping)

**Deployment strategy:**

- Rolling deployment: minimum 66% healthy, maximum 200%
- Deployment circuit breaker: automatically rolls back if new tasks fail health checks
- Health check grace period: 60 seconds (allows Python/model initialization)
- Deregistration delay: 120 seconds (allows in-flight LLM requests to complete)

**When to adjust:**

- If P99 latency is consistently high, increase `min_count`
- If tasks are being scaled out frequently, increase base CPU/memory per task
- For cost optimization, use Fargate Spot for tasks beyond the `min_count` baseline

### Redis Scaling

Redis scaling is vertical (bigger nodes) rather than horizontal:

- Scale up when `EngineCPUUtilization > 65%` sustained
- Scale up when `DatabaseMemoryUsagePercentage > 80%`
- Add read replicas if read throughput is the bottleneck (rare for this workload)

### Connection Draining

The ALB deregistration delay (120s) ensures in-flight requests complete during deployments:

1. ECS marks old task for draining
2. ALB stops sending new requests to the old task
3. Old task has 120s to finish in-flight requests
4. After 120s (or all connections close), old task is terminated

This is critical for streaming responses, which can take 10-60 seconds for LLM generation.

---

## Monitoring in Production

### CloudWatch (Default)

The Terraform configuration includes 7 CloudWatch alarms:

| Alarm | Metric | Threshold | Rationale |
|---|---|---|---|
| ALB 5xx Rate | HTTPCode_Target_5XX_Count | > 10/min | Spike in backend errors |
| P99 Latency | TargetResponseTime p99 | > 10s | LLM calls are slow; 10s is the ceiling |
| Unhealthy Targets | UnHealthyHostCount | > 0 | Any failed task is worth investigating |
| ECS CPU | CPUUtilization | > 85% | Scaling should have engaged before this |
| ECS Memory | MemoryUtilization | > 85% | Python can be memory-hungry; catch leaks |
| Redis CPU | EngineCPUUtilization | > 75% | Single-threaded engine approaching saturation |
| Redis Memory | DatabaseMemoryUsagePercentage | > 80% | Need headroom for LRU eviction |

**Pros:** Zero infrastructure to manage, native AWS integration, pay per metric, 15-month retention.

**Cons:** Limited custom dashboards, no native histogram/heatmap support, per-metric cost adds up.

### Self-Hosted Prometheus + Grafana (Optional)

The gateway already exposes a `/metrics/` endpoint with rich Prometheus metrics. To keep the existing Grafana dashboards (built in Phases 10 and 16):

1. Deploy Prometheus + Grafana as additional ECS tasks
2. Configure Prometheus to scrape gateway tasks via ECS service discovery
3. Import existing dashboard JSON files from `grafana/provisioning/dashboards/`
4. Use EBS volumes for Prometheus data persistence

**Pros:** Rich dashboards already built (4 dashboards), histogram/heatmap support, no per-metric cost, full control.

**Cons:** Additional ECS tasks (~$40-60/month), requires maintenance, storage management.

### Recommendation

**Start with CloudWatch alarms for operational alerts** (already configured in Terraform). Add self-hosted Prometheus + Grafana only if:

- You need the existing rich dashboards (streaming analytics, per-backend drilldown)
- Per-metric CloudWatch costs become significant (> 50 custom metrics)
- You need histogram-level analysis (P50/P95/P99 with full distribution visibility)

A hybrid approach works well: CloudWatch for paging/alerting, Prometheus for deep observability.

---

## Deployment Workflow

### Initial Deployment

```bash
# 1. Build and push image to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build --target runtime -t inference-gateway .
docker tag inference-gateway:latest <ecr-repo-url>:v1.0.0
docker push <ecr-repo-url>:v1.0.0

# 2. Configure Terraform
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values

# 3. Deploy
terraform init
terraform plan -out=tfplan
terraform apply tfplan

# 4. Verify
curl -s http://$(terraform output -raw alb_dns_name)/health
```

### Rolling Update

```bash
# Build and push new image
docker build --target runtime -t inference-gateway .
docker tag inference-gateway:latest <ecr-repo-url>:v1.1.0
docker push <ecr-repo-url>:v1.1.0

# Update task definition with new image tag
terraform apply -var="gateway_image_tag=v1.1.0"

# Monitor deployment
aws ecs describe-services --cluster inference-gateway-cluster --services inference-gateway-gateway
```
