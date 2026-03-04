.PHONY: build up down restart slam rviz shell clean help

help:
	@echo "GMM-SLAM Docker Helper"
	@echo "======================"
	@echo "make build    - Build Docker image"
	@echo "make up       - Start container"
	@echo "make down     - Stop container"
	@echo "make restart  - Restart container"
	@echo "make slam     - Run GMM-SLAM node"
	@echo "make rviz     - Launch RViz visualization"
	@echo "make shell    - Open bash shell in container"
	@echo "make clean    - Remove container and image"

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

restart: down up

slam:
	docker exec -it disal_slam run_gmmslam

rviz:
	docker exec -it disal_slam run_rviz

shell:
	docker exec -it disal_slam bash

clean:
	docker compose down -v
	docker rmi $$(docker images -q gmmslam-slam) || true
