.PHONY: up down test logs build seed status chaos chaos-down loadtest rolling-restart terraform-validate test-cli

up:
	docker compose up --build -d

down:
	docker compose down

test:
	docker build --target test -t inference-gateway:test .
	docker run --rm inference-gateway:test

logs:
	docker compose logs -f

build:
	docker compose build

seed:
	@bash scripts/seed.sh

status:
	@bash scripts/status.sh

chaos:
	docker compose -f docker-compose.yaml -f docker-compose.chaos.yml up --build -d

chaos-down:
	docker compose -f docker-compose.yaml -f docker-compose.chaos.yml down

loadtest:
	docker run --rm --network host \
	  -v $(PWD)/tests/load:/mnt/locust \
	  locustio/locust -f /mnt/locust/locustfile.py \
	  --host http://localhost:8080 \
	  --headless -u 20 -r 5 -t 60s \
	  --html /mnt/locust/report.html

rolling-restart:
	@bash scripts/rolling-restart.sh --build

terraform-validate:
	@bash scripts/test-terraform.sh

test-cli:
	docker compose up --build -d
	@echo "Waiting for services to be ready..."
	@sleep 20
	docker build --target cli-test -t inference-gateway:cli-test .
	docker run --rm --network inference-gateway_default \
	  -e IGW_GATEWAY_URL=http://nginx:80 \
	  -e TENANT_ALPHA_KEY=test-alpha-key \
	  -e TENANT_BETA_KEY=test-beta-key \
	  inference-gateway:cli-test
