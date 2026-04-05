output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.main.dns_name
}

output "ecr_repository_url" {
  description = "URL of the ECR repository for gateway images"
  value       = aws_ecr_repository.gateway.repository_url
}

output "redis_endpoint" {
  description = "Primary endpoint address for ElastiCache Redis"
  value       = aws_elasticache_replication_group.main.primary_endpoint_address
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "Name of the ECS service"
  value       = aws_ecs_service.gateway.name
}

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group for gateway logs"
  value       = aws_cloudwatch_log_group.gateway.name
}

output "sns_alarm_topic_arn" {
  description = "ARN of the SNS topic for CloudWatch alarm notifications"
  value       = aws_sns_topic.alarms.arn
}
