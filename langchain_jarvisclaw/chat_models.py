"""ChatJarvisClaw — LangChain ChatModel with x402 USDC payment support."""

from __future__ import annotations

from typing import Any, Optional

import httpx
from langchain_openai import ChatOpenAI


class ChatJarvisClaw(ChatOpenAI):
    """LangChain ChatModel for JarvisClaw AI API.

    Supports two authentication modes:
    1. API Key (traditional): pass `api_key="sk-..."`
    2. x402 Wallet Payment: pass `wallet_private_key="0x..."` to pay per-request with USDC

    x402 mode uses the `jarvisclaw` Python SDK under the hood, which handles
    the full payment flow: request → 402 → sign EIP-3009 USDC transfer → retry.

    Examples:
        # Mode 1: API Key (pre-paid balance)
        chat = ChatJarvisClaw(api_key="sk-xxx", model="gpt-5.4")

        # Mode 2: x402 (pay-per-request, no account needed)
        chat = ChatJarvisClaw(wallet_private_key="0x...", model="gpt-5.4")

        # Use like any LangChain ChatModel
        response = chat.invoke("Explain quantum computing")
    """

    base_url: str = "https://api.jarvisclaw.ai/v1"
    wallet_private_key: Optional[str] = None
    network: str = "base"  # "base" or "solana"
    _x402_max_retries: int = 2

    class Config:
        extra = "allow"

    def __init__(self, **kwargs: Any) -> None:
        # If using x402 mode, set a dummy API key (OpenAI client requires one)
        if kwargs.get("wallet_private_key") and not kwargs.get("api_key"):
            kwargs["api_key"] = "x402-wallet-payment"
        if "base_url" not in kwargs:
            kwargs["base_url"] = "https://api.jarvisclaw.ai/v1"
        super().__init__(**kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Override to handle x402 payment mode via jarvisclaw SDK."""
        if not self.wallet_private_key:
            # Standard API key mode — just call parent
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        # x402 mode: delegate to jarvisclaw SDK which handles the full
        # 402 → sign → retry flow with proper EIP-3009 USDC authorization
        from jarvisclaw import Client as JCClient

        jc = JCClient(private_key=self.wallet_private_key, chain=self.network)

        # Convert LangChain messages to OpenAI format
        formatted_messages = self._convert_messages(messages)

        response = jc.chat.completions.create(
            model=self.model_name,
            messages=formatted_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
            **{k: v for k, v in kwargs.items() if k not in ("stop", "run_manager")},
        )

        # Convert jarvisclaw SDK response back to LangChain format
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        content = response.choices[0].message.content or ""
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])

    def _convert_messages(self, messages) -> list[dict]:
        """Convert LangChain message objects to OpenAI dict format."""
        result = []
        for msg in messages:
            if hasattr(msg, "type"):
                role_map = {"human": "user", "ai": "assistant", "system": "system"}
                role = role_map.get(msg.type, msg.type)
            else:
                role = "user"
            result.append({"role": role, "content": msg.content})
        return result

    # ─── Convenience Methods ─────────────────────────────────────────────

    @classmethod
    def list_models(cls, base_url: str = "https://api.jarvisclaw.ai/v1") -> list[dict]:
        """List available models with pricing (no auth required).

        Returns:
            List of models with their pricing information.
        """
        response = httpx.get(f"{base_url.rstrip('/').replace('/v1', '')}/api/discovery/models", timeout=10.0)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])

    @classmethod
    def free_models(cls, base_url: str = "https://api.jarvisclaw.ai/v1") -> dict:
        """Get free and cheap models (no auth required).

        Returns:
            Dict with 'free' and 'cheap' model lists.
        """
        response = httpx.get(f"{base_url.rstrip('/').replace('/v1', '')}/api/discovery/free-models", timeout=10.0)
        response.raise_for_status()
        return response.json()

    @classmethod
    def health(cls, base_url: str = "https://api.jarvisclaw.ai/v1") -> dict:
        """Check platform health status (no auth required)."""
        response = httpx.get(f"{base_url.rstrip('/').replace('/v1', '')}/api/discovery/health", timeout=10.0)
        response.raise_for_status()
        return response.json()
