# FitVerse AI Service

This service submits the user's front, right, back, and left photos to Meshy's multi-image-to-3D API, polls task status, and streams completed GLB and thumbnail artifacts to the backend.

## Run locally

Set `AI_SERVICE_TOKEN` and `MESHY_API_KEY` in `backend/.env`, then from `backend/` run:

```powershell
docker compose up -d ai-service
```

The backend calls the service at `http://localhost:8001`. The service accepts calls only when the bearer token matches `AI_SERVICE_TOKEN`.

## Provider details

The service uses Meshy's `multi-image-to-3d` endpoint and requests an A-pose GLB. Provider generation requires Meshy API credits. The resulting mesh is not guaranteed to be rigged or suitable for measurement-accurate virtual try-on.

API reference: https://docs.meshy.ai/en/api/multi-image-to-3d
