import base64
import os
import secrets
from typing import Any, Dict, Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

PHOTO_ORDER = ("front", "right", "back", "left")
MAX_IMAGE_BYTES = 8 * 1024 * 1024
HTTP_TIMEOUT = httpx.Timeout(60.0, connect=10.0)

app = FastAPI(title="FitVerse Avatar Generation", version="0.1.0")


class GenerationInput(BaseModel):
    images: Dict[str, str]


def require_service_token(authorization: Optional[str] = Header(default=None)) -> None:
    expected_token = os.getenv("AI_SERVICE_TOKEN", "")
    if not expected_token:
        raise HTTPException(status_code=503, detail="AI service authentication is not configured")

    supplied_token = authorization.removeprefix("Bearer ") if authorization else ""
    if not secrets.compare_digest(supplied_token, expected_token):
        raise HTTPException(status_code=401, detail="Invalid service token")


def get_meshy_config() -> tuple[str, str, str]:
    api_key = os.getenv("MESHY_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="Meshy API key is not configured")

    base_url = os.getenv("MESHY_API_BASE_URL", "https://api.meshy.ai/openapi/v1").rstrip("/")
    model = os.getenv("MESHY_MODEL", "meshy-7.1")
    return api_key, base_url, model


def validate_image_data_uri(data_uri: str) -> None:
    header, separator, encoded_image = data_uri.partition(",")
    if not separator or header not in ("data:image/jpeg;base64", "data:image/png;base64"):
        raise HTTPException(status_code=400, detail="Photos must be JPEG or PNG base64 data URIs")

    try:
        image_bytes = base64.b64decode(encoded_image, validate=True)
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Photo data is not valid base64") from error

    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Each photo must be between 1 byte and 8 MB")


async def request_meshy(method: str, path: str, **kwargs: Any) -> httpx.Response:
    api_key, base_url, _ = get_meshy_config()
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = "Bearer " + api_key

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.request(method, base_url + path, headers=headers, **kwargs)
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail="Could not reach the Meshy API") from error

    if response.is_error:
        raise HTTPException(status_code=502, detail="Meshy API request failed")
    return response


async def get_meshy_task(task_id: str) -> Dict[str, Any]:
    response = await request_meshy("GET", "/multi-image-to-3d/" + task_id)
    return response.json()


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "fitverse-ai-service"}


@app.post("/v1/generations", dependencies=[Depends(require_service_token)])
async def create_generation(request: GenerationInput) -> Dict[str, str]:
    if set(request.images) != set(PHOTO_ORDER):
        raise HTTPException(status_code=400, detail="Exactly front, right, back, and left photos are required")

    for photo_type in PHOTO_ORDER:
        validate_image_data_uri(request.images[photo_type])

    _, _, model = get_meshy_config()
    response = await request_meshy(
        "POST",
        "/multi-image-to-3d",
        json={
            "image_urls": [request.images[photo_type] for photo_type in PHOTO_ORDER],
            "ai_model": model,
            "pose_mode": "a-pose",
            "should_texture": True,
            "target_formats": ["glb"],
        },
    )
    task_id = response.json().get("result")
    if not isinstance(task_id, str) or not task_id:
        raise HTTPException(status_code=502, detail="Meshy returned no generation task ID")

    return {"taskId": task_id, "status": "processing"}


@app.get("/v1/generations/{task_id}", dependencies=[Depends(require_service_token)])
async def generation_status(task_id: str) -> Dict[str, Any]:
    task = await get_meshy_task(task_id)
    provider_status = task.get("status")
    status = {
        "PENDING": "processing",
        "IN_PROGRESS": "processing",
        "SUCCEEDED": "completed",
        "FAILED": "failed",
        "CANCELED": "failed",
    }.get(provider_status)

    if status is None:
        raise HTTPException(status_code=502, detail="Meshy returned an unknown task status")
    if status == "completed" and not task.get("model_urls", {}).get("glb"):
        status = "failed"

    return {
        "taskId": task_id,
        "status": status,
        "progress": task.get("progress", 0),
        "error": task.get("task_error", {}).get("message") if status == "failed" else None,
    }


@app.get("/v1/generations/{task_id}/artifacts/{artifact_name}", dependencies=[Depends(require_service_token)])
async def generation_artifact(task_id: str, artifact_name: str) -> StreamingResponse:
    task = await get_meshy_task(task_id)
    if task.get("status") != "SUCCEEDED":
        raise HTTPException(status_code=409, detail="Generation is not complete")

    if artifact_name == "model":
        asset_url = task.get("model_urls", {}).get("glb")
        media_type = "model/gltf-binary"
        filename = "avatar.glb"
    elif artifact_name == "thumbnail":
        asset_url = task.get("thumbnail_url")
        media_type = "image/png"
        filename = "avatar.png"
    else:
        raise HTTPException(status_code=404, detail="Artifact not found")

    if not asset_url:
        raise HTTPException(status_code=404, detail="Generated artifact is unavailable")

    async def stream_asset():
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                async with client.stream("GET", asset_url) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        yield chunk
        except httpx.HTTPError as error:
            raise HTTPException(status_code=502, detail="Could not download generated artifact") from error

    return StreamingResponse(
        stream_asset(),
        media_type=media_type,
        headers={"Content-Disposition": "attachment; filename=" + filename},
    )