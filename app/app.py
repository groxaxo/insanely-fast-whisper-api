import os
from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Body,
    BackgroundTasks,
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
from typing import Optional

admin_key = os.environ.get("ADMIN_KEY")

hf_token = os.environ.get("HF_TOKEN")

# fly runtime env https://fly.io/docs/machines/runtime-environment
fly_machine_id = os.environ.get("FLY_MACHINE_ID")

whisper_device_env = os.environ.get("WHISPER_DEVICE")
if whisper_device_env is None:
    if torch.cuda.is_available():
        whisper_device_env = "cuda:0"
    else:
        whisper_device_env = "cpu"

if whisper_device_env.startswith("cuda"):
    torch.cuda.set_per_process_memory_fraction(0.6, 0)

use_flash_attn = os.environ.get("USE_FLASH_ATTENTION", "true").lower() == "true"

model_kwargs = {}
if use_flash_attn:
    model_kwargs["attn_implementation"] = "flash_attention_2"
    model_kwargs["load_in_4bit"] = True

# When using 4-bit quantization, don't specify device as it's handled by accelerate
if model_kwargs.get("load_in_4bit"):
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype=torch.float16 if whisper_device_env.startswith("cuda") else torch.float32,
        model_kwargs=model_kwargs,
    )
else:
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype=torch.float16 if whisper_device_env.startswith("cuda") else torch.float32,
        device=whisper_device_env,
        model_kwargs=model_kwargs,
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
        return HTTPException(status_code=400, detail="Not a background task")
    elif task.done() is False:
        task.cancel()
        del running_tasks[task_id]
        return {"status": "cancelled"}
    else:
        return {"status": "completed", "output": task.result()}


class TranscriptionRequest(BaseModel):
    file: Optional[str] = None  # In our implementation, we expect URL for the file
    model: str
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = "json"
    temperature: Optional[float] = 0.0


class TranslationRequest(BaseModel):
    file: Optional[str] = None  # In our implementation, we expect URL for the file
    model: str
    prompt: Optional[str] = None
    response_format: Optional[str] = "json"
    temperature: Optional[float] = 0.0


@app.post("/v1/audio/transcriptions")
def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
):
    """
    OpenAI-compatible transcription endpoint for Open WebUI
    """
    import tempfile
    import os
    
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        tmp_file.write(file.file.read())
        temp_file_path = tmp_file.name

    try:
        # Process the file using the existing logic
        outputs = process(
            temp_file_path,
            task="transcribe",
            language=language if language else "None",
            batch_size=64,
            timestamp="chunk",
            diarise_audio=False
        )
        
        # Format response based on response_format parameter
        if response_format == "text":
            return outputs.get("text", "")
        elif response_format in ["srt", "vtt"]:
            # For SRT/VTT formats, we need to convert the output accordingly
            # For now, return JSON format as fallback
            return outputs
        else:  # Default to JSON
            return outputs
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/v1/audio/translations")
def translate_audio(
    file: UploadFile = File(...),
    model: str = Form(...),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
):
    """
    OpenAI-compatible translation endpoint for Open WebUI
    """
    import tempfile
    import os
    
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        tmp_file.write(file.file.read())
        temp_file_path = tmp_file.name

    try:
        # Process the file using the existing logic
        outputs = process(
            temp_file_path,
            task="translate",
            language="None",  # Translation doesn't require specific language
            batch_size=64,
            timestamp="chunk",
            diarise_audio=False
        )
        
        # Format response based on response_format parameter
        if response_format == "text":
            return outputs.get("text", "")
        elif response_format in ["srt", "vtt"]:
            # For SRT/VTT formats, we need to convert the output accordingly
            # For now, return JSON format as fallback
            return outputs
        else:  # Default to JSON
            return outputs
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/v1/audio/translations")
def translate_audio(
    file: UploadFile = File(...),
    model: str = Form(...),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
):
    """
    OpenAI-compatible translation endpoint for Open WebUI
    """
    # Save uploaded file temporarily
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        tmp_file.write(file.file.read())
        temp_file_path = tmp_file.name

    try:
        # Process the file using the existing logic
        outputs = process(
            temp_file_path,
            task="translate",
            language="None",  # Translation doesn't require specific language
            batch_size=64,
            timestamp="chunk",
            diarise_audio=False
        )
        
        # Format response based on response_format parameter
        if response_format == "text":
            return outputs["text"]
        elif response_format in ["srt", "vtt"]:
            # For SRT/VTT formats, we need to convert the output accordingly
            # For now, return JSON format as fallback
            return outputs
        else:  # Default to JSON
            return outputs
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)