"""ChatJarvisClaw — LangChain ChatModel with x402 USDC payment support."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Optional

import httpx
from langchain_openai import ChatOpenAI

# Lazy imports for wallet signing (only needed in x402 mode)
_eth_account = None


def _get_eth_account():
    global _eth_account
    if _eth_account is None:
        from eth_account import Account
        from eth_account.messages import encode_defunct

        _eth_account = (Account, encode_defunct)
    return _eth_account


class ChatJarvisClaw(ChatOpenAI):
    """LangChain ChatModel for JarvisClaw AI API.

    Supports two authentication modes:
    1. API Key (traditional): pass `api_key="sk-..."`
    2. x402 Wallet Payment: pass `wallet_private_key="0x..."` to pay per-request with USDC

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
    network: str = "eip155:8453"  # Base mainnet
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
        """Override to handle 402 Payment Required responses."""
        if not self.wallet_private_key:
            # Standard API key mode — just call parent
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        # x402 mode: attempt request, handle 402, sign payment, retry
        for attempt in range(self._x402_max_retries + 1):
            try:
                return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as e:
                if not self._is_402_error(e) or attempt >= self._x402_max_retries:
                    raise
                # Extract 402 payment requirements and sign
                payment_info = self._extract_402_info(e)
                if payment_info:
                    signature = self._sign_x402_payment(payment_info)
                    # Inject payment signature into headers for next attempt
                    self._inject_payment_header(signature)

        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _is_402_error(self, error: Exception) -> bool:
        """Check if the error is an HTTP 402 Payment Required."""
        error_str = str(error)
        return "402" in error_str or "Payment Required" in error_str

    def _extract_402_info(self, error: Exception) -> Optional[dict]:
        """Extract payment requirements from a 402 response."""
        # Try to get the response body from the error
        try:
            # Make a direct HTTP request to get the full 402 response
            url = f"{self.base_url}/chat/completions"
            response = httpx.post(
                url,
                json={"model": self.model_name, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
                timeout=10.0,
            )
            if response.status_code == 402:
                import json
                return json.loads(response.text)
        except Exception:
            pass
        return None

    def _sign_x402_payment(self, payment_info: dict) -> str:
        """Sign an x402 payment using the wallet private key."""
        Account, encode_defunct = _get_eth_account()

        accepts = payment_info.get("accepts", [])
        if not accepts:
            raise ValueError("No payment options in 402 response")

        # Pick first matching network
        payment_option = None
        for opt in accepts:
            if opt.get("network") == self.network:
                payment_option = opt
                break
        if not payment_option:
            payment_option = accepts[0]

        # Build the payment message to sign
        amount = payment_option.get("amount", "0")
        pay_to = payment_option.get("payTo", "")
        resource_url = payment_info.get("resource", {}).get("url", "")

        # x402 payment signature: sign(amount + payTo + resource + timestamp)
        timestamp = str(int(time.time()))
        message_hash = hashlib.sha256(
            f"{amount}:{pay_to}:{resource_url}:{timestamp}".encode()
        ).hexdigest()

        msg = encode_defunct(text=message_hash)
        account = Account.from_key(self.wallet_private_key)
        signed = account.sign_message(msg)

        # Return formatted x402 payment signature
        import json
        return json.dumps({
            "scheme": payment_option.get("scheme", "exact"),
            "network": payment_option.get("network", self.network),
            "amount": amount,
            "payTo": pay_to,
            "signature": signed.signature.hex(),
            "signer": account.address,
            "timestamp": timestamp,
        })

    def _inject_payment_header(self, signature: str) -> None:
        """Inject PAYMENT-SIGNATURE header into the HTTP client."""
        if hasattr(self, "client") and self.client:
            # Access the underlying httpx client
            if hasattr(self.client, "_client"):
                self.client._client.headers["PAYMENT-SIGNATURE"] = signature
            self.default_headers = {
                **(self.default_headers or {}),
                "PAYMENT-SIGNATURE": signature,
            }

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
