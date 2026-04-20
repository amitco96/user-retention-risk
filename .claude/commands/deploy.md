---
description: Deploy to AWS ECS Fargate. Requires manual confirmation. DO NOT auto-trigger.
disable-model-invocation: true
allowed-tools: Bash, Read
---

# Production Deploy

⚠️  This command deploys to PRODUCTION. Confirm before proceeding.

## Pre-flight checklist
1. Run /run-tests — must pass
2. Run /review — must be APPROVED
3. Confirm: "Are you sure you want to deploy to production? (yes/no)"

If not confirmed, STOP.

## Deploy steps (only after confirmation)
1. Build Docker image:
   ```bash
   docker build -t user-retention-risk:$(git rev-parse --short HEAD) .
   ```
2. Push to ECR:
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_URI
   docker tag user-retention-risk:$(git rev-parse --short HEAD) $ECR_URI:$(git rev-parse --short HEAD)
   docker push $ECR_URI:$(git rev-parse --short HEAD)
   ```
3. Update ECS service:
   ```bash
   aws ecs update-service --cluster retention-cluster --service retention-service --force-new-deployment
   ```
4. Wait for stable:
   ```bash
   aws ecs wait services-stable --cluster retention-cluster --services retention-service
   ```
5. Tail logs for 60 seconds to confirm healthy startup
6. Hit /health endpoint on the live URL and confirm {"status": "ok"}
