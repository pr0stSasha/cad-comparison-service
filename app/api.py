"""HTTP API: приём PNG + описания, возврат отчёта о расхождениях."""
from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agent import DrawingAgent
from app.config import settings
from app.llm.client import LLMError
from app.pipeline.preprocess import ImageDecodeError

app = FastAPI(
    title="sheetgen-agent",
    description="Агент, описывающий различия между 3D-чертежом и его текстовым описанием",
    version="0.1.0",
)

agent = DrawingAgent()


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model": settings.llm_model,
        "llm_base_url": settings.llm_base_url,
        "vision": settings.llm_supports_vision,
    }


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(..., description="PNG-чертёж"),
    description: str = Form(..., description="Текстовое описание детали"),
) -> JSONResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Получен пустой файл")
    if not description.strip():
        raise HTTPException(status_code=400, detail="Описание детали не должно быть пустым")

    try:
        result = await agent.analyze(data, description)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return JSONResponse(result)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
