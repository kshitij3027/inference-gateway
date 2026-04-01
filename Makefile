.PHONY: up down test logs build seed status chaos chaos-down

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
