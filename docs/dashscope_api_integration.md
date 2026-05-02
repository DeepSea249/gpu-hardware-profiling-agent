# Alibaba Cloud DashScope / GLM-5 API Integration (Development Phase)

This document records how the Alibaba Cloud DashScope API (specifically the **GLM-5** model) was used during the development and testing phase of this project.  The production code has since been refactored to be fully OpenAI-API-compatible, but the development-phase configuration is preserved here for reference.

---

## 1. Background

During development the project used **Alibaba Cloud Bailian / DashScope** as the LLM back-end because:

- GLM-5 supported a proprietary *thinking-mode* (`enable_thinking`) that made multi-step hardware analysis more reliable during early development.
- DashScope provides an OpenAI-compatible REST endpoint, allowing the project to use the standard `openai` Python library with only a `base_url` override.

---

## 2. Endpoint Configuration

| Parameter | Development Value |
|---|---|
| Base URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| Model | `glm-5` |
| Auth header | `Authorization: Bearer <DASHSCOPE_API_KEY>` |
| SDK | `openai` Python package (v1.x) with `base_url` override |

The client was instantiated as:

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```

The API key was stored in a `.env` file at the project root and loaded via `python-dotenv`:

```
# .env (development)
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 3. DashScope-Specific Extensions Used

### 3.1 `enable_thinking` (thinking mode)

GLM-5 supports a proprietary body extension that activates an internal Chain-of-Thought reasoning pass before generating the final answer.  It was passed via `extra_body`:

```python
stream = client.chat.completions.create(
    model="glm-5",
    messages=messages,
    extra_body={"enable_thinking": True},   # DashScope-only
    stream=True,
    stream_options={"include_usage": True},
)
```

**Effect**: The model first emits a hidden `reasoning_content` delta stream (the thinking trace), then emits the final `content` delta stream (the answer).  This separation required a custom streaming loop that distinguished between the two.

### 3.2 `delta.reasoning_content`

Standard OpenAI streaming responses only carry `delta.content`.  With `enable_thinking` active, DashScope responses also carry `delta.reasoning_content` for the thinking trace.  The streaming loop used `hasattr` guards to handle both cases:

```python
for chunk in stream:
    if not chunk.choices:
        continue                          # final usage-only chunk
    delta = chunk.choices[0].delta

    # Thinking trace (DashScope only)
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        reasoning_buf += delta.reasoning_content

    # Final answer (standard OpenAI field)
    if hasattr(delta, "content") and delta.content:
        answer_buf += delta.content
```

`thinking_mode` was enabled selectively:

| Call site | `enable_thinking` | Rationale |
|---|---|---|
| `ReasoningEngine` (anomaly / synthesis) | `True` | Multi-step hardware reasoning benefits from CoT |
| `HardwareProber` (semantic resolution) | `False` | Single classification call, ~100 ms latency required |
| `ProbeCodeGenerator` (CUDA codegen) | `False` | Detailed specs make CoT unnecessary; 10× faster |

---

## 4. Retry Strategy

Transient failures (rate limits, timeouts, connection drops) were retried automatically with exponential back-off using `tenacity`:

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APITimeoutError, APIConnectionError, RateLimitError

@retry(
    retry=retry_if_exception_type((APITimeoutError, APIConnectionError, RateLimitError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    reraise=True,
)
def generate_reasoning(self, system_prompt, user_prompt): ...
```

---

## 5. LLM Call Sites

Four modules made LLM calls during development; each used a **lazy module-level singleton** so the LLMClient was only instantiated when actually needed and never crashed the agent if the API was unavailable:

```
src/reasoning.py        — anomaly analysis, methodology narratives, final synthesis
src/hardware_prober.py  — semantic target-name → probe-name resolution
src/kernel_analyzer.py  — kernel bottleneck report generation
src/probe_codegen.py    — autonomous CUDA micro-benchmark source generation
```

Each module defined:

```python
_llm_client = None

def _get_llm():
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    try:
        from .llm_client import LLMClient
        _llm_client = LLMClient(enable_thinking=False)
    except Exception as exc:
        logger.warning("LLM unavailable: %s", exc)
        _llm_client = None
    return _llm_client
```

Every call site had a graceful fallback to a template string so the agent could still produce results without an API key.

---

## 6. Migration to OpenAI-Compatible Interface

After the development phase was complete, the client was refactored to be fully OpenAI-API-compatible so that the final evaluation could use the **GPT-5.4** API without any code changes:

| | Development (DashScope) | Production (OpenAI-compatible) |
|---|---|---|
| Auth env var | `DASHSCOPE_API_KEY` | `API_KEY` (primary, eval); `DASHSCOPE_API_KEY` as dev fallback when `API_KEY` is absent |
| Base URL | Hardcoded DashScope URL | `BASE_URL` env var (defaults to `https://api.openai.com/v1`) |
| Model | Hardcoded `glm-5` | `BASE_MODEL` env var (eval default: `gpt-4o`; dev fallback: `glm-5`) |
| `extra_body` | Always sent | Only sent when `dashscope.aliyuncs.com` appears in the base URL |
| `delta.reasoning_content` | Always read | Only accessed on DashScope endpoint |

The detection logic in `llm_client.py`:

```python
_DASHSCOPE_HOST = "dashscope.aliyuncs.com"

class LLMClient:
    def __init__(self, ...):
        # Eval mode (primary): API_KEY + BASE_URL + BASE_MODEL
        # Dev mode (fallback): DASHSCOPE_API_KEY → auto-detect DashScope URL + glm-5
        self._base_url = base_url or os.getenv("BASE_URL", "https://api.openai.com/v1")
        self._is_dashscope = _DASHSCOPE_HOST in self._base_url

    def generate_reasoning(self, system_prompt, user_prompt):
        kwargs = dict(model=self.model, messages=messages, stream=True, ...)
        if self._is_dashscope:
            kwargs["extra_body"] = {"enable_thinking": self.enable_thinking}
        stream = self._client.chat.completions.create(**kwargs)
        for chunk in stream:
            delta = chunk.choices[0].delta
            if self._is_dashscope and hasattr(delta, "reasoning_content"):
                ...  # consume thinking trace
            if delta.content:
                answer_buf += delta.content
```

To run in DashScope/GLM-5 dev mode from any machine, simply set:

```
API_KEY=sk-xxxx
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
BASE_MODEL=glm-5
```

Or — on a machine where only `DASHSCOPE_API_KEY` is set (and `API_KEY` is absent), the
client auto-detects DashScope and selects `glm-5` without any additional configuration.
