# 🧠 **AI SRE Agent**
**Autonomous Incident Response and Root Cause Analysis using Generative AI**  

I built an AI-powered Site Reliability Engineering (SRE) assistant that detects, analyzes, and resolves production incidents automatically — turning hours of debugging into minutes.

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Google Cloud](https://img.shields.io/badge/GCP-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-000000?style=for-the-badge&logo=chainlink&logoColor=white)
![Made with 💙 by Prithvi Elancherran](https://img.shields.io/badge/Made%20with%20💙%20by-Prithvi%20Elancherran-blue?style=for-the-badge)

---

## 🎬 Demo Video

https://github.com/user-attachments/assets/a441064b-6982-44dc-86fc-a60b111379d0


## 📖 Overview

When production goes down at scale, chaos usually follows — pagers go off, multiple engineers join emergency calls, dashboards flood with alerts, and people start guessing.

> I wanted to change that.

**AI SRE Agent** is an **autonomous incident response system** I built that uses **Large Language Models (LLMs)**, **observability data**, and **historical incident intelligence** to perform **automated root cause analysis**, **playbook execution**, and **confidence-based recommendations**.

---

## ✈️ Inspiration

During my internship, I noticed how real-world incident response looked —  
> Pagers beeping, engineers scrambling, long debugging sessions, and uncertainty.

That experience motivated me to build an agent that could act as an **SRE co-pilot** — capable of analyzing observability signals, replaying past playbooks, and recommending the best actions.

The result was **AI SRE Agent** — an intelligent assistant that already knows *where to look first* when something breaks.

---

## 💡 Problem & Solution

### 🚨 Problem
Finding the root cause of a production issue can take hours:
- Dozens of systems to check  
- Logs and metrics spread across tools  
- Repeated incidents with no centralized memory  

### ⚙️ My Solution
I built an agent that:
- Analyzes incidents autonomously using observability data  
- Correlates current symptoms with historical patterns using vector similarity search  
- Executes troubleshooting playbooks automatically  
- Reports root causes and recommended actions with confidence scores  

---

## ⚙️ Key Features

| Category | Description |
|-----------|--------------|
| 🔍 **Incident Analysis** | AI-driven correlation with historical incidents |
| ⚙️ **Playbook Execution** | Automated troubleshooting workflow engine |
| 🧠 **Root Cause Detection** | Embedding-based similarity matching for diagnostics |
| 💬 **AI Chat Interface** | Real-time assistant for incident analysis commands |
| 📊 **Observability Integration** | Connects with GCP Monitoring, Logging, Error Reporting, and Tracing |
| 📈 **Confidence Scoring** | Quantifies reliability of AI inferences |
| 👩‍💻 **Human-in-the-Loop** | Allows engineers to approve high-impact actions |

---

## 🧪 Demo Scenarios

### 🎯 Scenario 1 — Payment API Latency Spike
- **Trigger:** Payment API latency crosses threshold  
- **Action:** The agent queries observability data → finds similar past incidents → surfaces root cause and recommendations  
- **Outcome:** Root cause identified (load imbalance) with **75.5% confidence**  
- **Recommendation:** Add request retries and rebalance load distribution  

### ⚙️ Scenario 2 — Database Connection Timeout
- **Trigger:** Billing service database connections nearing capacity  
- **Action:** Executes `database-timeout` playbook → analyzes metrics → detects connection pool saturation  
- **Outcome:** Root cause confirmed (pool capacity exceeded) with **53% confidence**  
- **Recommendation:** Scale connection pool or enable automatic retries  

---

## 🧰 Tech Stack

| Layer | Technologies |
|-------|---------------|
| **Frontend** | React 18, TypeScript, Vite, Tailwind CSS |
| **Backend** | FastAPI, Python 3.11, SQLAlchemy, Pydantic |
| **AI/ML** | LangChain (Agent Logic), Sentence Transformers (Embeddings), FAISS (Vector Search) |
| **Infra & DevOps** | Docker, Kubernetes (GKE), Terraform, Redis (Caching), GitHub Actions (CI/CD), GCP APIs |
| **Database** | SQLite (demo) / AlloyDB (production) |
| **Observability & Integrations** | GCP Monitoring, Logging, Tracing, Error Reporting APIs |
| **Security & Monitoring** | JWT Auth, OpenTelemetry, Prometheus Metrics |

---

## 🏗️ System Architecture

```
+-----------------------------+
|        Frontend (React)     |
|  - Chat Interface           |
|  - Incident Dashboard       |
|  - WebSocket Communication  |
+-------------+---------------+
              |
              | WebSocket / REST API
              |
+-------------v---------------+
|        Backend (FastAPI)    |
|  - Incident Analysis Engine |
|  - Playbook Executor        |
|  - Confidence Scoring       |
|  - GCP Integrations         |
+-------------+---------------+
              |
              v
+-----------------------------+
| GCP Monitoring + Logging    |
| Observability + Tracing     |
+-----------------------------+
```

---

## 📁 Project Structure

```
ai-sre-agent/
├── backend/
│   ├── api/              # API endpoints
│   ├── models/           # Schemas and data models
│   ├── services/         # Core AI and playbook logic
│   ├── integrations/     # GCP observability connectors
│   ├── utils/            # Helper functions
│   └── tests/            # Unit and integration tests
├── frontend/
│   ├── components/       # UI components
│   ├── services/         # API + WebSocket services
│   ├── hooks/            # Custom React hooks
│   └── types/            # TypeScript definitions
├── infrastructure/       # Docker, Terraform, Kubernetes
├── data/                 # Synthetic demo data
└── scripts/              # Setup and deployment scripts
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.11+  
- Node.js 18+  
- Docker & Docker Compose  
- Redis (for caching)

### 1️⃣ Clone & Setup
```bash
git clone <YOUR_REPOSITORY_URL>
cd ai-sre-agent
```

### 2️⃣ Start with Docker
```bash
chmod +x docker-*.sh
./docker-up.sh
```

### 3️⃣ Access the App
- **Frontend:** http://localhost:3000  
- **Backend API:** http://localhost:8000  
- **Docs:** http://localhost:8000/docs  

---

## 🧠 How It Works

1. An incident occurs → e.g., “Payment API latency spike”  
2. The AI SRE Agent retrieves logs, metrics, and traces  
3. Generates vector embeddings for current signals  
4. Searches historical incidents via similarity index  
5. Assigns confidence scores to findings  
6. Provides a clear root cause + recommended fix  
7. Optionally executes playbooks upon engineer approval  

---

## 📈 Impact

- ⏱️ Reduced **mean time to detect (MTTD)** and **mean time to resolve (MTTR)**  
- ⚙️ Automated repetitive incident triage workflows  
- 🧠 Reused knowledge from 100+ historical incidents  
- 🚀 Brought observability data, AI, and human-in-the-loop together  

---

## 🔐 Security & Reliability

- JWT-based authentication  
- Role-based access control  
- Pydantic validation for API inputs  
- CORS and rate limiting  
- Structured JSON logging with Loguru  
- OpenTelemetry tracing  

---

## 👨‍💻 What I Built

- Designed and implemented **AI-driven backend** using FastAPI and sentence transformers  
- Built **interactive frontend** in React for live incident chat and visualization  
- Integrated **GCP Monitoring, Logging, and Tracing APIs**  
- Implemented **confidence scoring system** for AI-generated diagnoses  
- Deployed **Dockerized microservices** on Google Kubernetes Engine (GKE)  
- Created **synthetic demo data** for local and cloud testing  

---

## 🖼️ Demo Snapshots

| Incident Dashboard | AI Chat Interface | Root Cause Analysis |
|--------------------|------------------|----------------------|
| ![Dashboard](docs/screenshots/dashboard.png) | ![Chat](docs/screenshots/chat.png) | ![Root Cause](docs/screenshots/rootcause.png) |

*(Replace paths with your uploaded screenshots)*

---

## 🔎 Example Commands

```bash
@sre-bot analyze incident INC-2024-001
@sre-bot execute playbook database-timeout
@sre-bot status
@sre-bot help
```

---

## 🧩 API Highlights

| Endpoint | Description |
|-----------|--------------|
| `POST /api/v1/incidents/analyze` | Run incident analysis |
| `GET /api/v1/incidents/{id}/timeline` | Fetch incident timeline |
| `POST /api/v1/playbooks/{id}/execute` | Execute troubleshooting playbook |
| `GET /api/v1/gcp/metrics` | Query GCP metrics |
| `WebSocket /ws/chat` | Real-time AI chat interface |

---

## 🧰 Development & Testing

```bash
# Run backend tests
cd backend
pytest tests/

# Lint and format
black backend/
flake8 backend/
mypy backend/
```

---

## 📦 Deployment

### Docker Compose
```bash
docker-compose up --build
```

### Kubernetes (GKE)
```bash
kubectl apply -f infrastructure/kubernetes/
```

### Terraform
```bash
cd infrastructure/terraform
terraform apply
```

---

## 🧭 Monitoring

- **Health Check:** `/health`  
- **Metrics:** Prometheus metrics (port 9090)  
- **Logs:** Structured JSON logs  
- **Tracing:** OpenTelemetry integration  

---

⭐ **AI SRE Agent — Designed & Built with 💙 by [Prithvi Elancherran](https://github.com/PrithviElancherran)**
