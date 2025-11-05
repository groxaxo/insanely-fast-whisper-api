# Insanely Fast Whisper API (Open WebUI Compatible)

An API to transcribe audio with [OpenAI's Whisper Large v3](https://huggingface.co/openai/whisper-large-v3)! Powered by 🤗 Transformers, Optimum & flash-attn. Now fully compatible with Open WebUI!

Features:
* 🎤 Transcribe audio to text at blazing fast speeds
* 📖 Fully open source and deployable on any GPU cloud provider
* 🗣️ Built-in speaker diarization
* ⚡ Easy to use and Fast API layer
* 📃 Async background tasks and webhooks
* 🔥 Optimized for concurrency and parallel processing
* ✅ Task management, cancel and status endpoints
* 🔒 Admin authentication for secure API access
* 🧩 Fully managed API available on [JigsawStack](https://jigsawstack.com/speech-to-text)
* 🌐 **NEW**: OpenAI-compatible endpoints for seamless Open WebUI integration

Based on [Insanely Fast Whisper CLI](https://github.com/Vaibhavs10/insanely-fast-whisper) project. Check it out if you like to set up this project locally or understand the background of insanely-fast-whisper.

This project is focused on providing a deployable blazing fast whisper API with docker on cloud infrastructure with GPUs for scalable production use cases.

## Docker image
```
yoeven/insanely-fast-whisper-api:latest
```
Docker hub: [yoeven/insanely-fast-whisper-api](https://hub.docker.com/r/yoeven/insanely-fast-whisper-api)

## Deployment
This application is dockerized and can be deployed on any cloud provider that supports docker and GPUs with a few config tweaks.

## Configuration
You can configure the application using environment variables:
- `ADMIN_KEY`: Admin API key for authentication (optional)
- `HF_TOKEN`: Hugging Face token for speaker diarization (optional)
- `WHISPER_DEVICE`: Device to run Whisper on (default: "cuda:0" if CUDA available, else "cpu")
- `USE_FLASH_ATTENTION`: Enable/disable Flash Attention 2 (default: "true")

To get the Hugging face token for speaker diarization you need to do the following:
1. Accept [`pyannote/segmentation-3.0`](https://hf.co/pyannote/segmentation-3.0) user conditions
2. Accept [`pyannote/speaker-diarization-3.1`](https://hf.co/pyannote/speaker-diarization-3.1) user conditions
3. Create an access token at [`hf.co/settings/tokens`](https://hf.co/settings/tokens).

## Fully managed and scalable API 
[JigsawStack](https://jigsawstack.com) provides a bunch of powerful APIs for various use cases while keeping costs low. This project is available as a fully managed API [here](https://jigsawstack.com/speech-to-text) with enhanced cloud scalability for cost efficiency and high uptime. Sign up [here](https://jigsawstack.com) for free!


## API usage

### Authentication
If you had set up the `ADMIN_KEY` environment variable. You'll need to pass `x-admin-api-key` in the header with the value of the key you previously set.

### Endpoints

#### Original Endpoints

#### **POST** `/`
Transcribe or translate audio into text
##### Body params (JSON)
| Name    | value |
|------------------|------------------|
| url (Required) |  url of audio |
| task | `transcribe`, `translate`  default: `transcribe` |
| language | `None`, `en`, [other languages](https://huggingface.co/openai/whisper-large-v3) default: `None` Auto detects language
| batch_size | Number of parallel batches you want to compute. Reduce if you face OOMs. default: `64` |
| timestamp | `chunk`, `word`  default: `chunk` |
| diarise_audio | Diarise the audio clips by speaker. You will need to set hf_token. default:`false` |
| webhook | Webhook `POST` call on completion or error. default: `None` |
| webhook.url | URL to send the webhook |
| webhook.header | Headers to send with the webhook |
| is_async | Run task in background and sends results to webhook URL. `true`, `false` default: `false` |
| managed_task_id | Custom Task ID used to reference ongoing task. default: `uuid() v4 will be generated for each transcription task` |

#### **GET** `/tasks`
Get all active transcription tasks, both async background tasks and ongoing tasks

#### **GET** `/status/{task_id}`
Get the status of a task, completed tasks will be removed from the list which may throw an error

#### **DELETE** `/cancel/{task_id}`
Cancel async background task. Only transcription jobs created with `is_async` set to `true` can be cancelled.

#### OpenAI-Compatible Endpoints (for Open WebUI)

#### **POST** `/v1/audio/transcriptions`
Transcribe audio into text in OpenAI-compatible format for use with Open WebUI

##### Form parameters:
- `file`: (Required) Audio file to transcribe
- `model`: (Required) Model to use (typically "whisper-1")
- `language`: (Optional) Language of the audio
- `prompt`: (Optional) An optional text to guide the model's style
- `response_format`: (Optional) Response format: `json` (default), `text`, `srt`, `vtt`, `verbose_json`
- `temperature`: (Optional) Temperature for sampling (default: 0.0)

#### **POST** `/v1/audio/translations`
Translate audio into English in OpenAI-compatible format for use with Open WebUI

##### Form parameters:
- `file`: (Required) Audio file to translate
- `model`: (Required) Model to use (typically "whisper-1") 
- `prompt`: (Optional) An optional text to guide the model's style
- `response_format`: (Optional) Response format: `json` (default), `text`, `srt`, `vtt`, `verbose_json`
- `temperature`: (Optional) Temperature for sampling (default: 0.0)

## Model Configuration

The API uses OpenAI's Whisper Large v3 model (`openai/whisper-large-v3`) with the following optimizations:
- 4-bit quantization for reduced memory usage
- Flash Attention 2 for efficient computation
- Float16 precision for reduced memory requirements
- Memory fraction limiting to 60% of GPU memory
- Batch processing for improved throughput

## Memory Optimization

To achieve minimal VRAM usage, the following parameters are used:
- `USE_FLASH_ATTENTION=true` - Enables Flash Attention 2 for memory-efficient attention computation
- `WHISPER_DEVICE=cuda:0` - Specifies the CUDA device to use
- `torch.cuda.set_per_process_memory_fraction(0.6)` - Limits GPU memory usage to 60% of available memory
- `load_in_4bit=True` - Uses 4-bit quantization to reduce model memory footprint
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` - Configures PyTorch memory allocation to prevent fragmentation

## Running locally with pip/venv
```bash
# clone the repo
$ git clone https://github.com/jigsawstack/insanely-fast-whisper-api.git

# change the working directory
$ cd insanely-fast-whisper-api

# install torch
$ pip3 install torch torchvision torchaudio

# upgrade wheel and install required packages for FlashAttention
$ pip3 install -U wheel && pip install ninja packaging

# install FlashAttention
$ pip3 install flash-attn --no-build-isolation

# generate updated requirements.txt if you want to use other management tools (Optional)
$ poetry export --output requirements.txt

# get the path of python
$ which python3

# setup virtual environment 
$ poetry env use /full/path/to/python

# install the requirements
$ poetry install

# run the app
$ uvicorn app.app:app --reload
```

## Running locally with Conda (Recommended for GPU)

For easier GPU setup and dependency management, we recommend using Conda. See [README-Conda.md](README-Conda.md) for detailed instructions on installing and running the API with Conda.

A quick installation script is also available:
```bash
# Run the automated installation script
./install_conda.sh
```

## Performance Benchmarks

Here are some benchmarks we ran on Nvidia A100 - 80GB GPU infra👇
| Optimization type    | Time to Transcribe (150 mins of Audio) |
|------------------|------------------|
| **large-v3 (Transformers) (`fp16` + `batching [24]` + `Flash Attention 2`)** | **~2 (*1 min 38 sec*)**            |
| **large-v3 (Transformers) (`fp16` + `batching [24]` + `Flash Attention 2` + `diarization`)** | **~2 (*3 min 16 sec*)**            |
| **large-v3 (Transformers) (`fp16` + `batching [24]` + `Flash Attention 2`)** | **~2 (*1 min 58 sec*)**            |
| **large-v3 (Transformers) (`fp16` + `batching [24]` + `Flash Attention 2` + `diarization`)** | **~2 (*3 min 36 sec*)**|

The estimated startup time for the machine with GPU and loading up the model is around ~20 seconds. The rest of the time is spent on the actual computation.

## Extra
The application is designed to be efficient and cost-effective. The application will optimize resource usage based on your hardware configuration.

## Acknowledgements

1. [Vaibhav Srivastav](https://github.com/Vaibhavs10) for writing a huge chunk of the code and the CLI version of this project.
2. [OpenAI Whisper](https://huggingface.co/openai/whisper-large-v3) 

## JigsawStack
This project is part of [JigsawStack](https://jigsawstack.com) - A suite of powerful and developer friendly APIs for various use cases while keeping costs low. Sign up [here](https://jigsawstack.com) for free!