# langchain-jarvisclaw

LangChain integration for [JarvisClaw](https://jarvisclaw.ai) AI API — 280+ models, pay-per-request with USDC via x402.

## Install

```bash
pip install langchain-jarvisclaw
```

> **Wallet mode requires 0.1.1 or later.** In 0.1.0, passing `wallet_private_key`
> raised `ImportError` — it imported a name the jarvisclaw SDK does not export, so the
> x402 path never ran. API-key mode was unaffected. If you are pinned to 0.1.0 and
> using a wallet, upgrade.

## Usage

### Mode 1: API Key (pre-paid balance)

```python
from langchain_jarvisclaw import ChatJarvisClaw

chat = ChatJarvisClaw(api_key="sk-...", model="gpt-5.4")
response = chat.invoke("Explain quantum computing in one sentence")
print(response.content)
```

### Mode 2: x402 Wallet Payment (no account needed)

Pay per request with USDC on Base. No signup, no API key — just a wallet.

```python
from langchain_jarvisclaw import ChatJarvisClaw

chat = ChatJarvisClaw(
    wallet_private_key="0x...",  # Your EVM wallet private key
    model="anthropic/claude-sonnet-4.6",
)
response = chat.invoke("Write a haiku about AI agents")
print(response.content)
```

### Discover Models & Pricing (no auth)

```python
from langchain_jarvisclaw import ChatJarvisClaw

# List all models with USD pricing
models = ChatJarvisClaw.list_models()
for m in models[:5]:
    print(f"{m['model']}: ${m['input_per_m_token_usd']}/M input tokens")

# Find free models to test quality
free = ChatJarvisClaw.free_models()
print(f"Free models: {[m['model'] for m in free['free']]}")

# Check platform health
health = ChatJarvisClaw.health()
print(f"Status: {health['status']}, Models: {health['data']['total_models']}")
```

### Use with LangChain chains

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_jarvisclaw import ChatJarvisClaw

chat = ChatJarvisClaw(api_key="sk-...", model="gpt-5.4")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding assistant."),
    ("human", "{question}"),
])

chain = prompt | chat
response = chain.invoke({"question": "How do I reverse a linked list in Go?"})
```

### Streaming

```python
chat = ChatJarvisClaw(api_key="sk-...", model="gpt-5.4", streaming=True)

for chunk in chat.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

## Supported Models

40+ models including:
- OpenAI: gpt-5.5, gpt-5.4, gpt-5.4-mini, o3, o1
- Anthropic: claude-opus-4.7, claude-sonnet-4.6, claude-haiku-4.5
- Google: gemini-3.1-pro, gemini-2.5-flash
- DeepSeek: deepseek-chat, deepseek-reasoner
- And many more — call `ChatJarvisClaw.list_models()` for the full list

## Payment Networks

- **Base (EVM)**: USDC on Base mainnet (default)
- **Solana**: USDC on Solana (install with `pip install langchain-jarvisclaw[solana]`)

## License

MIT
