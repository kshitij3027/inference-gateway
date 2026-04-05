# -----------------------------------------------------------------------------
# General
# -----------------------------------------------------------------------------

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used for resource naming and tagging"
  type        = string
  default     = "inference-gateway"
}

variable "environment" {
  description = "Deployment environment (e.g. production, staging)"
  type        = string
  default     = "production"
}

variable "tags" {
  description = "Default tags applied to all resources"
  type        = map(string)
  default = {
    Project   = "inference-gateway"
    ManagedBy = "Terraform"
  }
}

# -----------------------------------------------------------------------------
# Networking
# -----------------------------------------------------------------------------

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "az_count" {
  description = "Number of availability zones to use"
  type        = number
  default     = 2
}

variable "enable_nat_gateway" {
  description = "Whether to create NAT Gateway(s) for private subnet internet access"
  type        = bool
  default     = true
}

variable "single_nat_gateway" {
  description = "Use a single NAT Gateway instead of one per AZ (cost savings for non-prod)"
  type        = bool
  default     = true
}

variable "certificate_arn" {
  description = "ACM certificate ARN for HTTPS listener (leave empty to skip HTTPS)"
  type        = string
  default     = ""
}

# -----------------------------------------------------------------------------
# ECS / Gateway
# -----------------------------------------------------------------------------

variable "gateway_image_tag" {
  description = "Docker image tag for the gateway container"
  type        = string
  default     = "latest"
}

variable "gateway_cpu" {
  description = "CPU units for gateway task (1024 = 1 vCPU)"
  type        = number
  default     = 1024
}

variable "gateway_memory" {
  description = "Memory (MiB) for gateway task"
  type        = number
  default     = 2048
}

variable "gateway_desired_count" {
  description = "Desired number of gateway tasks"
  type        = number
  default     = 3
}

variable "gateway_min_count" {
  description = "Minimum number of gateway tasks for autoscaling"
  type        = number
  default     = 2
}

variable "gateway_max_count" {
  description = "Maximum number of gateway tasks for autoscaling"
  type        = number
  default     = 10
}

variable "gateway_port" {
  description = "Port the gateway container listens on"
  type        = number
  default     = 8080
}

# -----------------------------------------------------------------------------
# Redis / ElastiCache
# -----------------------------------------------------------------------------

variable "redis_node_type" {
  description = "ElastiCache node instance type"
  type        = string
  default     = "cache.t4g.micro"
}

variable "redis_num_cache_clusters" {
  description = "Number of cache clusters (nodes) in the replication group"
  type        = number
  default     = 2
}

# -----------------------------------------------------------------------------
# Observability
# -----------------------------------------------------------------------------

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

# -----------------------------------------------------------------------------
# Gateway Environment Variables
# -----------------------------------------------------------------------------

variable "config_path" {
  description = "Path to backend configuration file inside the container"
  type        = string
  default     = "config/backends.yaml"
}

variable "log_level" {
  description = "Application log level"
  type        = string
  default     = "info"
}

variable "cache_ttl" {
  description = "Cache TTL in seconds"
  type        = number
  default     = 300
}

variable "cache_similarity_threshold" {
  description = "Semantic cache similarity threshold (0.0–1.0)"
  type        = string
  default     = "0.95"
}

variable "queue_max_depth" {
  description = "Maximum request queue depth"
  type        = number
  default     = 100
}

variable "queue_timeout" {
  description = "Queue timeout in seconds"
  type        = number
  default     = 30
}

variable "l1_max_entries" {
  description = "L1 in-memory cache max entries"
  type        = number
  default     = 500
}

variable "l1_ttl" {
  description = "L1 in-memory cache TTL in seconds"
  type        = number
  default     = 60
}
