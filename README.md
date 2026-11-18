# Real-Time-Financial-Fraud-Detection-Platform
A reproducible fraud-scoring platform that replays transactions through Kafka, blends XGBoost with account graph signals, and presents analyst feedback in Streamlit.

## Quick start

```bash
make setup
cp .env.example .env
docker compose -f deploy/docker-compose.yml up -d
make eda
make features
make train
make serve
```

The IEEE-CIS dataset is the preferred input. If Kaggle access is unavailable, place a
PaySim CSV at `data/raw/paysim.csv`; the loaders and splitters support both schemas.

## Validation targets

- Primary metric: time-based hold-out PR-AUC (not accuracy).
- API p95 target: under 100 ms per transaction on a laptop.
- Feature parity: Feast offline and Redis online values agree within tolerance.
- Deployment: Minikube preferred, Docker Compose fallback.

See `docs/architecture.md`, `docs/api.md`, `docs/data.md`, and `docs/PROGRESS.md`.
