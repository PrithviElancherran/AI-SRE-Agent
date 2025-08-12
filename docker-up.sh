#!/bin/bash

echo "Starting AI SRE Agent with Docker Compose..."

# Build and start services
docker compose up --build -d

echo ""
echo "Services starting up..."
echo "Waiting for services to be healthy..."

# Wait for backend to be healthy
echo -n "Backend: "
while ! docker compose exec backend curl -f http://localhost:8000/health > /dev/null 2>&1; do
    echo -n "."
    sleep 2
done
echo " ✓ Ready"

# Check frontend
echo -n "Frontend: "
while ! curl -f http://localhost:3000 > /dev/null 2>&1; do
    echo -n "."
    sleep 2
done
echo " ✓ Ready"

echo ""
echo "AI SRE Agent is running!"
echo "- Frontend: http://localhost:3000"
echo "- Backend API: http://localhost:8000"
echo "- API Docs: http://localhost:8000/docs"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop: ./docker-down.sh"