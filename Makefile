.PHONY: up down test logs build

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
