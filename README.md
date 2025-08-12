# AI SRE Agent

**Date:** July 10, 2025  

## Prerequisites
- **Docker and Docker Compose** installed
- **Python 3.11.13** installed
- **Git** installed

## Clone and Setup Project

```bash
# Clone the repository
git clone <YOUR_REPOSITORY_URL>
cd 4.0-ai-sre-agent (Root directory)
```

### Docker Setup to Run the Project:
```bash
Start the application:
./docker-up.sh
```

This single command will:
- Build both backend and frontend Docker images
- Start both services in the background
- Wait for them to be healthy
- Show you the URLs when ready

Access the application:
- `Frontend: http://localhost:3000`
- `Backend API: http://localhost:8000`
- `API Documentation: http://localhost:8000/docs`

View logs (optional, in a new terminal):
```bash
./docker-logs.sh
```

Stop the application and cleanup:
```bash
./docker-down.sh
```

#### First Time Setup:

If the scripts aren't executable, run this first:
```bash
chmod +x docker-*.sh
```

That's it! Just run ./docker-up.sh from the root directory and everything
will be set up automatically.

# AI SRE Agent Demo

AI SRE Agent is a demonstration of automated root cause analysis of production issues using AI, observability data, playbooks, and incident history.

## Features

- **Automated Incident Analysis**: AI-powered correlation with historical incidents
- **Playbook-Driven Debugging**: Systematic troubleshooting workflows
- **Real-time Chat Interface**: Interactive debugging and analysis
- **GCP Observability Integration**: Monitoring, logging, error reporting, and tracing
- **Vector Similarity Search**: Intelligent incident correlation
- **Confidence Scoring**: AI-driven confidence assessment
- **Human-in-the-Loop**: Safety guardrails for automated actions

## Demo Scenarios

### Scenario 1: Past Incident Correlation & Diagnosis
- **Trigger**: Payment API latency spike detection
- **Process**: Historical incident search → Verification → Resolution recommendation
- **Outcome**: Automated root cause analysis with confidence scoring

### Scenario 2: Playbook-Driven Debugging
- **Trigger**: Database connection timeout troubleshooting
- **Process**: Playbook selection → Step execution → Root cause identification
- **Outcome**: Systematic debugging with human approval workflow

## Architecture

- **Backend**: Python FastAPI with async/await support
- **Frontend**: React TypeScript with real-time WebSocket communication
- **Database**: SQLite for demo, AlloyDB for production
- **Caching**: Redis for GCP API response caching
- **Deployment**: Docker containers on Google Kubernetes Engine

## Technology Stack

- **Backend**: FastAPI, SQLAlchemy, Pydantic, sentence-transformers
- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS
- **Database**: SQLite (demo), AlloyDB (production)
- **Caching**: Redis
- **GCP Services**: Monitoring, Logging, Error Reporting, Tracing
- **AI/ML**: Sentence transformers for vector embeddings

## Project Structure

```
GenAI_CE_AI_SRE1/
├── backend/               # FastAPI backend application
│   ├── config/           # Configuration and settings
│   ├── models/           # Pydantic models and database schemas
│   ├── services/         # Business logic and AI services
│   ├── api/              # FastAPI routes and endpoints
│   ├── integrations/     # GCP observability integrations
│   ├── utils/            # Utility functions and helpers
│   └── tests/            # Unit and integration tests
├── frontend/             # React TypeScript frontend
│   └── src/
│       ├── components/   # React components
│       ├── services/     # API and WebSocket services
│       ├── types/        # TypeScript type definitions
│       └── hooks/        # Custom React hooks
├── data/                 # Synthetic data for demo
│   └── synthetic/        # Generated demo data
├── docs/                 # Documentation
├── infrastructure/       # Deployment configurations
│   ├── terraform/        # Infrastructure as Code
│   ├── kubernetes/       # Kubernetes manifests
│   └── docker/           # Docker configurations
└── scripts/              # Setup and deployment scripts
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (optional)
- Redis (for caching)

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Demo Usage

1. **Open the Chat Interface**: Navigate to the frontend application
2. **Trigger Scenario 1**: Type `@sre-bot: Payment API latency > 5s in US region`
3. **Trigger Scenario 2**: Type `@sre-bot: Database connection timeout errors in ecommerce-api`
4. **Observe Analysis**: Watch the AI agent analyze symptoms and correlate with historical data
5. **Review Recommendations**: Examine the suggested actions and confidence scores
6. **Approve Actions**: Use the human-in-the-loop approval workflow

## Configuration

Key configuration options in `backend/config/settings.py`:

- `DEMO_MODE`: Enable demo mode with synthetic data
- `SIMILARITY_THRESHOLD`: Incident correlation threshold
- `CONFIDENCE_THRESHOLD`: Automated action confidence threshold
- `GCP_PROJECT_ID`: Google Cloud Project ID
- `DATABASE_URL`: Database connection string

## API Endpoints

- `POST /api/v1/incidents/analyze`: Trigger incident analysis
- `GET /api/v1/incidents/{id}/timeline`: Get incident timeline
- `GET /api/v1/playbooks`: List available playbooks
- `POST /api/v1/playbooks/{id}/execute`: Execute playbook
- `GET /api/v1/gcp/metrics`: Query GCP monitoring metrics
- `WebSocket /ws/chat`: Real-time chat interface

## Development

### Running Tests

```bash
cd backend
pytest tests/
```

### Code Quality

```bash
black backend/
flake8 backend/
mypy backend/
```

### Frontend Development

```bash
cd frontend
npm run lint
npm run type-check
npm run test
```

## Deployment

### Docker Deployment

```bash
docker-compose up --build
```

### Kubernetes Deployment

```bash
kubectl apply -f infrastructure/kubernetes/
```

### GCP Deployment

```bash
# Deploy to Google Kubernetes Engine
gcloud container clusters create ai-sre-agent
kubectl apply -f infrastructure/kubernetes/
```

## Monitoring

- **Health Check**: `/health`
- **Metrics**: Prometheus metrics on port 9090
- **Logs**: Structured JSON logs with loguru
- **Tracing**: OpenTelemetry integration

## Security

- **Authentication**: JWT-based authentication
- **Authorization**: Role-based access control
- **Rate Limiting**: API request rate limiting
- **CORS**: Configurable CORS policies
- **Input Validation**: Pydantic model validation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support, please contact the SRE team or create an issue in the repository.
