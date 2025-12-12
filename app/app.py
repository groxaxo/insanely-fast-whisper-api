import os
import tempfile
from fastapi import (
    FastAPI,
    HTTPException,
    Body,
    Request,
    File,
    UploadFile,
    Form,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from transformers import pipeline
from .diarization_pipeline import diarize
import requests
import asyncio
import uuid


# Environment variables
admin_key = os.environ.get("ADMIN_KEY")
hf_token = os.environ.get("HF_TOKEN")

# Fly.io runtime environment variable
# https://fly.io/docs/machines/runtime-environment
fly_machine_id = os.environ.get("FLY_MACHINE_ID")

# Optional: Configure GPU memory limit
# Uncomment and adjust the fraction (e.g., 0.15 = 15%) to limit GPU memory usage
# if torch.cuda.is_available():
#     torch.cuda.set_per_process_memory_fraction(0.5, device=0)
#     torch.cuda.empty_cache()
#     print(f"GPU: {torch.cuda.get_device_name(0)}")
#     print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    torch_dtype=torch.float16,
    device="cuda:0",
    model_kwargs=({"attn_implementation": "flash_attention_2"}),
)

app = FastAPI()
loop = asyncio.get_event_loop()
running_tasks = {}


class WebhookBody(BaseModel):
    url: str
    header: dict[str, str] = {}


def process(
    url: str,
    task: str,
    language: str,
    batch_size: int,
    timestamp: str,
    diarise_audio: bool,
    webhook: WebhookBody | None = None,
    task_id: str | None = None,
):
    errorMessage: str | None = None
    outputs = {}
    try:
        generate_kwargs = {
            "task": task,
            "language": None if language == "None" else language,
        }

        outputs = pipe(
            url,
            chunk_length_s=30,
            batch_size=batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps="word" if timestamp == "word" else True,
        )

        if diarise_audio is True:
            speakers_transcript = diarize(
                hf_token,
                url,
                outputs,
            )
            outputs["speakers"] = speakers_transcript
    except asyncio.CancelledError:
        errorMessage = "Task Cancelled"
    except Exception as e:
        errorMessage = str(e)

    if task_id is not None:
        del running_tasks[task_id]

    if webhook is not None:
        webhookResp = (
            {"output": outputs, "status": "completed", "task_id": task_id}
            if errorMessage is None
            else {"error": errorMessage, "status": "error", "task_id": task_id}
        )

        if fly_machine_id is not None:
            webhookResp["fly_machine_id"] = fly_machine_id

        requests.post(
            webhook.url,
            headers=webhook.header,
            json=(webhookResp),
        )

    if errorMessage is not None:
        raise Exception(errorMessage)

    return outputs


@app.middleware("http")
async def admin_key_auth_check(request: Request, call_next):
    if admin_key is not None:
        if ("x-admin-api-key" not in request.headers) or (
            request.headers["x-admin-api-key"] != admin_key
        ):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    response = await call_next(request)
    return response


@app.post("/")
def root(
    url: str = Body(),
    task: str = Body(default="transcribe", enum=["transcribe", "translate"]),
    language: str = Body(default="None"),
    batch_size: int = Body(default=64),
    timestamp: str = Body(default="chunk", enum=["chunk", "word"]),
    diarise_audio: bool = Body(
        default=False,
    ),
    webhook: WebhookBody | None = None,
    is_async: bool = Body(default=False),
    managed_task_id: str | None = Body(default=None),
):
    if url.lower().startswith("http") is False:
        raise HTTPException(status_code=400, detail="Invalid URL")

    if diarise_audio is True and hf_token is None:
        raise HTTPException(status_code=500, detail="Missing Hugging Face Token")

    if is_async is True and webhook is None:
        raise HTTPException(
            status_code=400, detail="Webhook is required for async tasks"
        )

    task_id = managed_task_id if managed_task_id is not None else str(uuid.uuid4())

    try:
        resp = {}
        if is_async is True:
            backgroundTask = asyncio.ensure_future(
                loop.run_in_executor(
                    None,
                    process,
                    url,
                    task,
                    language,
                    batch_size,
                    timestamp,
                    diarise_audio,
                    webhook,
                    task_id,
                )
            )
            running_tasks[task_id] = backgroundTask
            resp = {
                "detail": "Task is being processed in the background",
                "status": "processing",
                "task_id": task_id,
            }
        else:
            running_tasks[task_id] = None
            outputs = process(
                url,
                task,
                language,
                batch_size,
                timestamp,
                diarise_audio,
                webhook,
                task_id,
            )
            resp = {
                "output": outputs,
                "status": "completed",
                "task_id": task_id,
            }
        if fly_machine_id is not None:
            resp["fly_machine_id"] = fly_machine_id
        return resp
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
def tasks():
    return {"tasks": list(running_tasks.keys())}


@app.get("/status/{task_id}")
def status(task_id: str):
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = running_tasks[task_id]

    if task is None:
        return {"status": "processing"}
    elif task.done() is False:
        return {"status": "processing"}
    else:
        return {"status": "completed", "output": task.result()}


@app.delete("/cancel/{task_id}")
def cancel(task_id: str):
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = running_tasks[task_id]
    if task is None:
        raise HTTPException(status_code=400, detail="Not a background task")
    elif task.done() is False:
        task.cancel()
        del running_tasks[task_id]
        return {"status": "cancelled"}
    else:
        return {"status": "completed", "output": task.result()}


# OpenAI-compatible endpoint for Open WebUI
@app.post("/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-large-v3-turbo"),
    language: str = Form(default=None),
    response_format: str = Form(default="json"),
):
    """
    OpenAI-compatible transcription endpoint for Open WebUI integration.
    Accepts audio file upload and returns transcription in OpenAI format.
    """
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Clear GPU cache before processing (if CUDA is available)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Process the audio file
            generate_kwargs = {
                "task": "transcribe",
                "language": language if language and language != "None" else None,
            }

            outputs = pipe(
                temp_file_path,
                chunk_length_s=30,
                batch_size=8,
                generate_kwargs=generate_kwargs,
                return_timestamps=True,
            )

            # Clear cache after processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Return in OpenAI-compatible format
            return {
                "text": outputs["text"]
            }

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    except Exception as e:
        print(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
