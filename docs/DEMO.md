# Demo runbook

The live demo requires a Docker-enabled GPU host, an NVIDIA container runtime,
and the local model artifacts mounted at `./models`.

```bash
cp .env.example .env
docker compose up --build
```

Wait for the API health endpoint:

```bash
curl http://localhost:8000/api/v1/health
```

Submit an invoice image and retain the returned `task_id`:

```bash
curl -X POST http://localhost:8000/api/v1/extract \
  -F "file=@invoice.jpg"
```

Poll until `status` becomes `success`:

```bash
curl http://localhost:8000/api/v1/tasks/<task_id>
```

The Streamlit interface is available at `http://localhost:8501`. Record the
upload, extraction progress, structured result, and invoice list views from
this real flow for the project demo video.
