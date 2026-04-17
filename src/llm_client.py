"""
LLM Client — OpenAI-compatible API integration.

Works with any OpenAI-compatible endpoint (OpenAI, Azure OpenAI,
Alibaba Cloud DashScope, etc.).  All configuration is via environment
variables so no credentials are ever hardcoded in source.

Environment variables
---------------------
API_KEY       (required) API key / access token.
LLM_BASE_URL  (optional) Base URL of the OpenAI-compatible endpoint.
              Default: https://api.openai.com/v1
LLM_MODEL     (optional) Model name.
              Default: gpt-4o

Usage:
    from src.llm_client import LLMClient

    client = LLMClient()  # reads API_KEY from environment
    answer = client.generate_reasoning(
        system_prompt="You are a GPU performance expert.",
        user_prompt="Explain shared memory bank conflicts.",
    )
"""

import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger("GPUAgent.LLMClient")

# Load .env from the project root (parent of src/)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)

# Transient exceptions that warrant a retry
_RETRYABLE = (APITimeoutError, APIConnectionError, RateLimitError)

# DashScope host identifier — used to enable DashScope-specific extensions
_DASHSCOPE_HOST = "dashscope.aliyuncs.com"


class LLMClient:
    """OpenAI-compatible LLM client with retry & streaming support.

    Configured entirely via environment variables:
        API_KEY       — API key (also accepts DASHSCOPE_API_KEY as legacy fallback)
        LLM_BASE_URL  — endpoint base URL (default: https://api.openai.com/v1)
        LLM_MODEL     — model name        (default: gpt-4o)
    """

    DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
    DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        enable_thinking: bool = True,
    ):
        # Primary env var is API_KEY; fall back to the legacy DashScope key
        resolved_key = (
            api_key
            or os.getenv("API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "No API key found. Set the API_KEY environment variable "
                "(or put it in .env / pass api_key= explicitly)."
            )

        self._base_url = base_url or self.DEFAULT_BASE_URL
        self._client = OpenAI(api_key=resolved_key, base_url=self._base_url)
        self.model = model or self.DEFAULT_MODEL
        # enable_thinking is a DashScope/GLM-specific extension; ignored elsewhere
        self.enable_thinking = enable_thinking
        # Detect DashScope endpoint to conditionally enable proprietary extensions
        self._is_dashscope = _DASHSCOPE_HOST in self._base_url

    # ------------------------------------------------------------------ #
    #  Core API – with automatic retry (max 3, exponential backoff 2-8s)  #
    # ------------------------------------------------------------------ #
    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def generate_reasoning(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Send a chat completion request and return the assistant's final
        answer as a plain string.

        Internally uses streaming so that the thinking / answer tokens
        are printed in real-time when log-level is DEBUG.

        Args:
            system_prompt: The system-level instruction.
            user_prompt:   The user query.

        Returns:
            The full answer text (excluding the hidden thinking trace).
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        # enable_thinking is a DashScope-specific body extension; omit for standard OpenAI
        if self._is_dashscope:
            kwargs["extra_body"] = {"enable_thinking": self.enable_thinking}

        stream = self._client.chat.completions.create(**kwargs)

        reasoning_content = ""
        answer_content = ""

        for chunk in stream:
            if not chunk.choices:
                # Final chunk carries token usage only
                if chunk.usage:
                    logger.debug("Token usage: %s", chunk.usage)
                continue

            delta = chunk.choices[0].delta

            # reasoning_content is a DashScope-specific delta attribute
            if self._is_dashscope and (
                hasattr(delta, "reasoning_content")
                and delta.reasoning_content is not None
            ):
                reasoning_content += delta.reasoning_content
                logger.debug("%s", delta.reasoning_content)

            if hasattr(delta, "content") and delta.content:
                answer_content += delta.content

        logger.info(
            "LLM call complete – reasoning: %d chars, answer: %d chars",
            len(reasoning_content),
            len(answer_content),
        )
        return answer_content

    # ------------------------------------------------------------------ #
    #  Convenience wrappers                                               #
    # ------------------------------------------------------------------ #
    def analyze_metrics(self, metrics_json: str) -> str:
        """Ask the LLM to interpret GPU profiling metrics."""
        return self.generate_reasoning(
            system_prompt=(
                "You are an expert GPU performance engineer. "
                "Analyze the following Nsight Compute metrics and identify "
                "the primary bottleneck. Be precise and cite metric names."
            ),
            user_prompt=metrics_json,
        )

    def explain_anomaly(self, anomaly_description: str) -> str:
        """Ask the LLM to explain a detected hardware anomaly."""
        return self.generate_reasoning(
            system_prompt=(
                "You are a GPU hardware expert. Explain the following "
                "anomaly detected during micro-benchmarking and suggest "
                "possible root causes."
            ),
            user_prompt=anomaly_description,
        )
