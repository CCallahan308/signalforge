# Deployment

SignalForge deploys as a single Streamlit app. There is no API, database, or background service.

## Streamlit Community Cloud (the live demo)

The public demo runs `src/app/dashboard_demo.py` — a static showcase that needs no trained
artifacts (the cloud host has none). Point a Streamlit Cloud app at this repo, set the main file
to `src/app/dashboard_demo.py`, and it installs `requirements.txt` automatically.

For the data-backed dashboard (`src/app/dashboard.py`), the trained artifacts in
`models/artifacts/` must be present — that requires running the pipeline, so it's best run
locally or in an environment where you can execute training.

## Docker (local, data-backed dashboard)

```bash
# 1) produce artifacts first (see docs/SETUP.md)
python scripts/generate_sample_data.py        # or download_real_data.py --source telco
python scripts/engineer_real_features.py
python scripts/train_with_optuna.py

# 2) build and run the dashboard
docker compose up --build                      # http://localhost:8501
```

`docker-compose.yml` mounts `./data` and `./models` so the container reads the artifacts you
generated on the host.

## Notes

- The image is pinned to `python:3.11-slim` to match the pinned scientific stack.
- No secrets are required at runtime. The only external credential (Kaggle, for data download)
  is used outside the container.
