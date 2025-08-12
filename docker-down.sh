#!/bin/bash

echo "Stopping AI SRE Agent..."

# Stop and remove containers
docker compose down

docker compose down -v # Remove volumes if needed

echo "AI SRE Agent stopped."