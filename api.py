import io
import json
import os
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fal_client
import numpy as np
import requests
from PIL import Image
from pydantic import BaseModel
from misc.config import load_config

import constants as con
from helper import log


cfg = load_config()
GIGACHAT_BASIC_AUTH = cfg.basic_auth
GIGACHAT_SCOPE = cfg.scope
GIGACHAT_TOKEN_URL = cfg.token_url
GIGACHAT_API_BASE = cfg.api_base
GIGACHAT_MODEL = cfg.model
GIGACHAT_EMBEDDING_MODEL = cfg.embedding_model
GIGACHAT_VERIFY = cfg.verify

_gigachat_token = None
_gigachat_token_expiry = None


# --- GitHub Models (OpenAI-совместимый REST) ---
# Бесплатно на gh-токене, доступно из РФ без VPN.
# В GitHub Actions хватает встроенного GITHUB_TOKEN при permissions: models: read.
GITHUB_MODELS_TOKEN = (
    os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or os.getenv("MODELS_TOKEN") or ""
)
GITHUB_MODELS_BASE = os.getenv(
    "GITHUB_MODELS_BASE", "https://models.github.ai/inference"
)
GITHUB_MODELS_MODEL = os.getenv("GITHUB_MODELS_MODEL", "openai/gpt-4o-mini")
# Провайдер по умолчанию для всех LLM-вызовов: "github" | "gigachat".
DEFAULT_LLM_API = os.getenv("LLM_API", "github")


def github_chat_completion(messages, model=None, temperature=0.5, max_tokens=2048):
    """Чат-комплишн через GitHub Models (OpenAI-совместимый эндпоинт)."""
    if not GITHUB_MODELS_TOKEN:
        raise RuntimeError("GITHUB_TOKEN/GH_TOKEN отсутствует для GitHub Models")
    url = f"{GITHUB_MODELS_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {GITHUB_MODELS_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model or GITHUB_MODELS_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_err = None
    for attempt in range(4):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 429:
                wait = 0
                try:
                    wait = int(response.headers.get("retry-after", "0"))
                except Exception:
                    wait = 0
                wait = wait or 5 * (attempt + 1)
                log(f"GitHub Models 429 (rate limit). Ждём {wait}s, попытка {attempt + 1}/4.")
                time.sleep(wait)
                continue
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            log(f"GitHub Models ошибка (попытка {attempt + 1}/4): {e}")
            time.sleep(3 * (attempt + 1))
    raise last_err


def _resolve_provider(api):
    """Выбираем эффективного провайдера. Если просят GigaChat, но он не
    сконфигурирован — откатываемся на GitHub Models."""
    provider = api or DEFAULT_LLM_API
    if provider == "gigachat" and (
        not GIGACHAT_BASIC_AUTH or GIGACHAT_BASIC_AUTH == "MISSING_BASIC_AUTH"
    ):
        provider = "github"
    return provider


def _resolve_model(model, provider):
    """Чиним модель под провайдера: id GitHub Models неймспейснутые (openai/...)."""
    if provider == "github":
        if model and "/" in model:
            return model
        return GITHUB_MODELS_MODEL
    return model or GIGACHAT_MODEL


class Article(BaseModel):
    desc: str
    title: str


class List(BaseModel):
    items: list[str]


class ArticleFull(BaseModel):
    desc: str
    emoji: str
    title: str


def get_gigachat_token():
    """fetch and cache gigachat oauth token"""
    global _gigachat_token, _gigachat_token_expiry

    now = datetime.now(timezone.utc)
    if not GIGACHAT_BASIC_AUTH or GIGACHAT_BASIC_AUTH == "MISSING_BASIC_AUTH":
        raise RuntimeError("GIGACHAT_BASIC_AUTH missing")
    if (
        _gigachat_token
        and _gigachat_token_expiry
        and now < _gigachat_token_expiry - timedelta(seconds=30)
    ):
        return _gigachat_token

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
        "Authorization": f"Basic {GIGACHAT_BASIC_AUTH}",
    }
    payload = {"scope": GIGACHAT_SCOPE}

    response = requests.post(
        GIGACHAT_TOKEN_URL,
        headers=headers,
        data=payload,
        timeout=30,
        verify=GIGACHAT_VERIFY,
    )
    response.raise_for_status()
    data = response.json()

    _gigachat_token = data.get("access_token")
    expires_at = data.get("expires_at") or data.get("expires_in")
    try:
        if isinstance(expires_at, (int, float)):
            seconds = expires_at / 1000.0 if expires_at > 10 * 365 * 24 * 3600 else expires_at
            _gigachat_token_expiry = now + timedelta(seconds=int(seconds))
        elif isinstance(expires_at, str):
            _gigachat_token_expiry = (
                datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                if "T" in expires_at
                else now + timedelta(minutes=30)
            )
        else:
            _gigachat_token_expiry = now + timedelta(minutes=30)
    except Exception:
        _gigachat_token_expiry = now + timedelta(minutes=30)

    return _gigachat_token


def gigachat_chat_completion(messages, model=GIGACHAT_MODEL, temperature=0.5, max_tokens=1024):
    token = get_gigachat_token()
    url = f"{GIGACHAT_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model or GIGACHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(
        url, headers=headers, json=payload, timeout=60, verify=GIGACHAT_VERIFY
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def gigachat_embedding(text, model=GIGACHAT_EMBEDDING_MODEL):
    token = get_gigachat_token()
    url = f"{GIGACHAT_API_BASE}/embeddings"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"model": model or GIGACHAT_EMBEDDING_MODEL, "input": [text]}

    response = requests.post(
        url, headers=headers, json=payload, timeout=60, verify=GIGACHAT_VERIFY
    )
    response.raise_for_status()
    data = response.json()
    return data["data"][0]["embedding"]


def get_json(
    prompt,
    api="gigachat",
    model=GIGACHAT_MODEL,
    temperature=0.0001,
    system_prompt="You are a helpful assistant.",
):
    text = get_text(
        prompt=prompt,
        system_prompt=system_prompt,
        api=api,
        model=model,
        temperature=temperature,
    )
    text = re.sub(r"```json|```python|```", "", text).strip()
    try:
        doc = json.loads(text)
    except:
        try:
            text = text.replace("'", '"')
            doc = json.loads(text)
        except:
            log(f"Error. Failed to parse JSON from LLM. {text}")
            doc = {"error": "Parsing error", "raw_data": text}
    return doc


def get_structured(
    prompt,
    cls,
    model=GIGACHAT_MODEL,
    temperature=0.0001,
    api="gigachat",
    system_prompt="You are a helpful assistant.",
):
    doc = {"error": "Parsing error"}
    schema = json.dumps(cls.model_json_schema(), ensure_ascii=False)
    structured_prompt = (
        f"{prompt}\n\nReturn JSON that matches this schema: {schema}. "
        "Respond with JSON only."
    )

    raw = get_text(
        prompt=structured_prompt,
        api=api,
        model=model,
        temperature=temperature,
        system_prompt=system_prompt,
    )
    try:
        doc = json.loads(raw)
        if isinstance(doc, dict):
            allowed_keys = set(cls.model_fields.keys())
            filtered = {k: v for k, v in doc.items() if k in allowed_keys}
            if filtered:
                doc = filtered
    except Exception as e:
        log(f"Error. Failed to parse JSON. Details: {e}. Response: {raw}")
        doc = {"error": "Parsing error", "raw_data": raw, "details": str(e)}

    return doc


def get_text(
    prompt,
    api="gigachat",
    model=GIGACHAT_MODEL,
    temperature=0.5,
    system_prompt="You are a helpful assistant.",
):
    provider = _resolve_provider(api)
    model = _resolve_model(model, provider)
    log(f"{provider} request. Model: {model}. Prompt: {prompt}")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    if provider == "github":
        text = github_chat_completion(
            messages, model=model, temperature=temperature, max_tokens=2048
        )
    else:
        text = gigachat_chat_completion(
            messages, model=model, temperature=temperature, max_tokens=2048 #1024
        )

    log(f"Response: {text}")
    if any([x in text for x in con.RENAME_TERMS.keys()]):
        log("Renaming some terms.")
        for k, v in con.RENAME_TERMS.items():
            text = text.replace(k, v)

    return text


def get_embedding(text, size=256):
    try:
        res = gigachat_embedding(text)[:size]
        res = normalize_l2(res)

        return res.tolist()
    except Exception as e:
        log(f"Error fetching embedding: {e}")
        return []


def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)


def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(log["message"])


def generate_and_save_image(name, img_dir, prompt):
    log(f"Generating image by prompt: {prompt}.")
    try:
        result = fal_client.subscribe(
            "fal-ai/flux/schnell",
            arguments={
                "prompt": prompt,
                "seed": 42,
                "image_size": {"width": 384, "height": 720},
                "num_images": 1,
            },
            with_logs=True,
            on_queue_update=on_queue_update,
        )
        img = result["images"][0]
        log(f'Saving generated image from {img["url"]} to {name}.')
        Path(img_dir).mkdir(exist_ok=True)
        image_path = os.path.join(img_dir, name)
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(img["url"], headers=headers)
        image = Image.open(io.BytesIO(response.content))
        output_io = io.BytesIO()
        image.save(output_io, format="JPEG", quality=60)
        output_io.seek(0)

        with open(image_path, "wb") as fout:
            fout.write(output_io.read())
    except Exception as e:
        log(f"Error generating an image: {e}")
        result = ""

    return result


def generate_image_for_paper(paper, img_name):
    title = paper["title"]
    abstract = paper["abstract"]
    prompt = f"Write a text with image prompt in style of surrealism and modern art based on the following paper. Use key themes and elements from it. Add instruction to write a text that reads as brief paper title as a label on some object on an image. Style: linear art on white background. Return only prompt and nothing else. Title: '{title}' Text: '{abstract}'"
    img_prompt = get_text(
        prompt, api="gigachat", model=GIGACHAT_MODEL, temperature=0.8
    )
    img_dir = os.path.join(con.IMG_DIR, paper["pub_date"].replace("-", ""))
    generate_and_save_image(name=img_name, img_dir=img_dir, prompt=img_prompt)


def get_categories(text, api="gigachat", model=GIGACHAT_MODEL):
    prompt_cls_1 = f"""Analyze the following research paper text and classify it into one or more relevant topics from the list below. Consider only information from the provided text. Don't add a tag if the topic is not directly related to the article.

Topics:

DATASET: Papers that introduce new datasets or make significant modifications to existing ones
DATA: Papers focusing on data processing, cleaning, collection, or curation methodologies
BENCHMARK: Papers proposing or analyzing model evaluation frameworks and benchmarks
AGENTS: Papers exploring autonomous agents, web agents, or agent-based architectures
RL: Papers investigating reinforcement learning theory or applications
RLHF: Papers specifically about human feedback in RL (PPO, DPO, etc.)
RAG: Papers advancing retrieval-augmented generation techniques
PLP: Papers about Programming Language Processing models or programming benchmarks
INFERENCE: Papers optimizing model deployment (quantization, pruning, etc.)
3D: Papers on 3D content generation, processing, or understanding
AUDIO: Papers advancing speech/audio processing or generation
MULTIMODAL: Papers combining multiple input/output modalities
MATH: Papers focused on mathematical theory and algorithms
MULTILINGUAL: Papers addressing multiple languages or cross-lingual capabilities, including all non English models
ARCHITECTURE: Papers proposing novel neural architectures or components
HEALTHCARE: Papers applying ML to medical/healthcare domains
TRAINING: Papers improving model training or fine-tuning methods
ROBOTICS: Papers on robotic systems and embodied AI
SMALL_MODELS: Papers that describe models considering small, below 1 billion parameters or similar
MEMORY: Works on internal/external memory for LLMs
AGENT_MEMORY: Memory and long-term context in agents
SESSION_MEMORY: Dialogue/session memory across turns
RETRIEVAL: Retrieval models and retrievers (distinct from full RAG)
RETRIEVAL_EVAL: Benchmarks/metrics for retrieval robustness
PROMPTING: Prompt engineering, CoT/ToT, self-consistency
PROMPT_CACHING: KV/response caching for long sessions
CONTEXT_WINDOW: Techniques to extend/pack context
COMPRESSION: Context compression/summarization/chunking for memory
MEMORIZATION: On-the-fly learning/continual memorization
CONTINUAL_LEARNING: Continual learning without forgetting
KNOWLEDGE_GRAPHS: Knowledge storage via graphs
TOOL_USE: Using external tools as memory/extension
OPEN_SOURCE: Papers that contribute to open-source projects
SCIENCE: Papers on scientific applications of ML
LOW_RESOURCE: Papers mentioning low-resource settings or languages

Return only a Python flat list of topics that match the given text.

Paper text to classify:\n\n"{text}"
"""
    
    prompt_cls_2 = f"""Analyze the following research paper text and classify it into one or more relevant topics from the list below. Consider only information from the provided text. Don't add a tag if the topic is not directly related to the article.

Topics:

AGI: Papers discussing artificial general intelligence concepts
GAMES: Papers applying ML to games or game development
INTERPRETABILITY: Papers analyzing model behavior and explanations
REASONING: Papers enhancing logical reasoning capabilities
TRANSFER_LEARNING: Papers on knowledge transfer between models/domains
GRAPHS: Papers advancing graph neural networks and applications
ETHICS: Papers addressing AI ethics, fairness, and bias
SECURITY: Papers on model security and adversarial robustness
OPTIMIZATION: Papers advancing training optimization methods
SURVEY: Papers comprehensively reviewing research areas
DIFFUSION: Papers on diffusion-based generative models
ALIGNMENT: Papers about aligning language models with human values, preferences, and intended behavior
STORY_GENERATION: Papers on story generation, including plot generation and author style adaptation
HALLUCINATIONS: Papers about the hallucinations, hallucinations analysis and mitigation
LONG_CONTEXT: Papers about long context handling, including techniques to extend context length
SYNTHETIC: Papers about using synthetic data for training, including methods for generating and leveraging artificial data
TRANSLATION: Papers on machine translation, including techniques, data and applications for translating between languages
LEAKAGE: Papers about data leakage, including issues of unintended data exposure and methods to detect or prevent it
OPEN_SOURCE: Papers that contribute to open-source projects by releasing models, datasets, or frameworks to the public
SCIENCE: Papers on scientific applications of LM including understanding of science articles and research automatization
LOW_RESOURCE: Papers that mention low-resource languages
MEMORY: Works on internal/external memory for LLMs
AGENT_MEMORY: Memory and long-term context in agents
SESSION_MEMORY: Dialogue/session memory across turns
RETRIEVAL: Retrieval models and retrievers (distinct from full RAG)
RETRIEVAL_EVAL: Benchmarks/metrics for retrieval robustness
PROMPTING: Prompt engineering, CoT/ToT, self-consistency
PROMPT_CACHING: KV/response caching for long sessions
CONTEXT_WINDOW: Techniques to extend/pack context
COMPRESSION: Context compression/summarization/chunking for memory
MEMORIZATION: On-the-fly learning/continual memorization
CONTINUAL_LEARNING: Continual learning without forgetting
KNOWLEDGE_GRAPHS: Knowledge storage via graphs
TOOL_USE: Using external tools as memory/extension

Return only a Python flat list of topics that match the given text.

Paper text to classify:\n\n"{text}"
"""

#     prompt_cls = f"""{{
#   "task": "Classify the following machine learning research paper into one or more relevant categories.",
#   "instructions": "Analyze the provided research paper text and classify it into one or more of the categories listed below. Focus on the paper's main contributions, methodologies, and applications.",
#   "categories": [
#     {{
#       "name": "DATASET",
#       "description": "Papers that introduce new datasets or significantly modify existing ones for research purposes."
#     }},
#     {{
#       "name": "DATA",
#       "description": "Papers focusing on methodologies for data processing, cleaning, collection, or curation."
#     }},
#     {{
#       "name": "BENCHMARK",
#       "description": "Papers proposing or analyzing frameworks for evaluating models or benchmarks."
#     }},
#     {{
#       "name": "AGENTS",
#       "description": "Papers exploring autonomous agents, web agents, or agent-based architectures."
#     }},
#     {{
#       "name": "CV",
#       "description": "Papers developing methods in computer vision or visual processing systems."
#     }},
#     {{
#       "name": "RL",
#       "description": "Papers investigating reinforcement learning theory or its applications."
#     }},
#     {{
#       "name": "RLHF",
#       "description": "Papers specifically about incorporating human feedback into reinforcement learning (e.g., PPO, DPO)."
#     }},
#     {{
#       "name": "RAG",
#       "description": "Papers advancing techniques for retrieval-augmented generation."
#     }},
#     {{
#       "name": "PLP",
#       "description": "Papers about programming language processing models or programming benchmarks."
#     }},
#     {{
#       "name": "INFERENCE",
#       "description": "Papers optimizing model deployment, including techniques like quantization or pruning."
#     }},
#     {{
#       "name": "3D",
#       "description": "Papers on 3D content generation, processing, or understanding."
#     }},
#     {{
#       "name": "AUDIO",
#       "description": "Papers advancing speech or audio processing or generation."
#     }},
#     {{
#       "name": "VIDEO",
#       "description": "Papers on video analysis, generation, or understanding."
#     }},
#     {{
#       "name": "MULTIMODAL",
#       "description": "Papers combining multiple input/output modalities."
#     }},
#     {{
#       "name": "MATH",
#       "description": "Papers focused on mathematical theory and algorithms in machine learning."
#     }},
#     {{
#       "name": "MULTILINGUAL",
#       "description": "Papers addressing multiple languages or cross-lingual capabilities, including non-English models."
#     }},
#     {{
#       "name": "ARCHITECTURE",
#       "description": "Papers proposing novel neural architectures or components."
#     }},
#     {{
#       "name": "MEDICINE",
#       "description": "Papers applying machine learning to medical or healthcare domains."
#     }},
#     {{
#       "name": "TRAINING",
#       "description": "Papers improving model training or fine-tuning methods."
#     }},
#     {{
#       "name": "ROBOTICS",
#       "description": "Papers on robotic systems and embodied AI."
#     }},
#     {{
#       "name": "AGI",
#       "description": "Papers discussing concepts related to artificial general intelligence."
#     }},
#     {{
#       "name": "GAMES",
#       "description": "Papers applying machine learning to games or game development."
#     }},
#     {{
#       "name": "INTERPRETABILITY",
#       "description": "Papers analyzing model behavior and providing explanations."
#     }},
#     {{
#       "name": "REASONING",
#       "description": "Papers enhancing logical reasoning capabilities in AI systems."
#     }},
#     {{
#       "name": "TRANSFER_LEARNING",
#       "description": "Papers on knowledge transfer between models or domains."
#     }},
#     {{
#       "name": "GRAPHS",
#       "description": "Papers advancing graph neural networks and their applications."
#     }},
#     {{
#       "name": "ETHICS",
#       "description": "Papers addressing AI ethics, fairness, and bias."
#     }},
#     {{
#       "name": "SECURITY",
#       "description": "Papers on model security and adversarial robustness."
#     }},
#     {{
#       "name": "EDGE_COMPUTING",
#       "description": "Papers on deploying machine learning models on resource-constrained devices."
#     }},
#     {{
#       "name": "OPTIMIZATION",
#       "description": "Papers advancing training optimization methods."
#     }},
#     {{
#       "name": "SURVEY",
#       "description": "Papers comprehensively reviewing research areas."
#     }},
#     {{
#       "name": "DIFFUSION",
#       "description": "Papers on diffusion-based generative models."
#     }},
#     {{
#       "name": "ALIGNMENT",
#       "description": "Papers about aligning language models with human values, preferences, and intended behavior."
#     }},
#     {{
#       "name": "STORY_GENERATION",
#       "description": "Papers on story generation, including plot generation and author style adaptation."
#     }},
#     {{
#       "name": "HALLUCINATIONS",
#       "description": "Papers about model hallucinations, their analysis, and mitigation strategies."
#     }},
#     {{
#       "name": "LONG_CONTEXT",
#       "description": "Papers about handling long contexts, including techniques to extend context length."
#     }},
#     {{
#       "name": "SYNTHETIC",
#       "description": "Papers about using synthetic data for training, including methods for generating and leveraging artificial data."
#     }},
#     {{
#       "name": "TRANSLATION",
#       "description": "Papers about machine translation, including techniques, data, and applications for translating between languages."
#     }},
#     {{
#       "name": "LEAKAGE",
#       "description": "Papers about data leakage, including issues of unintended data exposure and methods to detect or prevent it."
#     }},
#     {{
#       "name": "OPEN_SOURCE",
#       "description": "Papers that contribute to open-source projects by releasing models, datasets, or frameworks to the public."
#     }},
#     {{
#       "name": "MODELS",
#       "description": "Papers that describe machine learning models, whether they are open-source or proprietary."
#     }}
#   ],
#   "paper_text": "{text}",
#   "output_format": "Return only a Python flat list of categories that match the given text."
# }}"""
    categories_1 = get_json(
        prompt_cls_1, api=api, model=model, temperature=0.0
    )
    categories_2 = get_json(
        prompt_cls_2, api=api, model=model, temperature=0.0
    )

    # res = []
    # for c in categories:
    #     if c in CAT_MAPPING:
    #         res.append(CAT_MAPPING[c])

    # Make sure we always operate on lists, even if get_json returned a dict
    if isinstance(categories_1, str):
        categories_1 = [categories_1]
    if isinstance(categories_1, dict):
        categories_1 = categories_1.get("categories") or categories_1.get("topics") or []
    if isinstance(categories_2, str):
        categories_2 = [categories_2]
    if isinstance(categories_2, dict):
        categories_2 = categories_2.get("categories") or categories_2.get("topics") or []
    categories = list(set((categories_1 or []) + (categories_2 or [])))

    return categories


def get_categories_additional(text, api="gigachat", model=GIGACHAT_MODEL):
    prompt_cls = f"""You are an expert classifier of machine learning research papers. Analyze the following research paper text and classify it into one or more relevant categories from the list below.

Categories:
1. MULTILINGUAL: Papers addressing multiple languages or cross-lingual capabilities, including all non English models
2. LONG_CONTEXT: Papers about long context handling, including techniques to extend context length
3. SYNTHETIC: Papers about using synthetic data for training, including methods for generating and leveraging artificial data
4. TRANSLATION: Papers about machine translation, including techniques, data and applications for translating between languages
5. TRAINING: Papers about improving model training or fine-tuning methods, including optimization techniques, training strategies, and related methodologies

Return only JSON with flat array of categories that match the given text. If no category fit return empty list.

Paper text to classify:\n\n"{text}"
"""
    categories = get_json(
        prompt_cls, api=api, model=model, temperature=0.0
    )
    # Normalize possible dict outputs into a flat list
    if isinstance(categories, str):
        categories = [categories]
    if isinstance(categories, dict):
        for key in ("categories", "topics", "tags", "data", "result"):
            if key in categories and isinstance(categories[key], list):
                categories = categories[key]
                break
        else:
            categories = []
    elif not isinstance(categories, list):
        categories = []

    categories = [x for x in categories if x not in con.EXCLUDE_CATS]
    categories = [
        x if x not in con.RENAME_CATS else con.RENAME_CATS[x] for x in categories
    ]
    categories = [f"#{x.replace('#','')}".lower() for x in categories]

    return categories
