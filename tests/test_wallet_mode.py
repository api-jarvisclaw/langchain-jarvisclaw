"""The x402 wallet branch of ChatJarvisClaw._generate.

This branch never ran in a released version. It did:

    from jarvisclaw import Client as JCClient
    jc = JCClient(private_key=..., chain=...)

and neither name is real. The jarvisclaw SDK has never exported a bare ``Client``
(it is per-capability — ``ChatClient``, ``ImageClient``, … — plus an OpenAI-compatible
shim), and the constructor parameter is ``network``, not ``chain``. So passing
``wallet_private_key`` raised ImportError before any request was made, while the
API-key branch worked fine — which is why nobody noticed.

These tests never touch the network. They assert the names the wrapper reaches for
actually exist and are spelled the way the SDK spells them, which is the entire class
of defect that shipped.
"""

import inspect

import pytest

jarvisclaw = pytest.importorskip("jarvisclaw", reason="jarvisclaw SDK not installed")


# A syntactically valid Base key. Never used to sign — construction is the assertion.
DUMMY_EVM_KEY = "0x" + "11" * 32


def test_sdk_does_not_export_a_bare_Client():
    """Pin the fact that made the old import fail.

    If a future SDK adds a top-level ``Client``, this test fails and someone rereads
    this file — which is the point. It is a statement about why the code looks the way
    it does, not a wish for the SDK to stay still.
    """
    assert not hasattr(jarvisclaw, "Client"), (
        "jarvisclaw now exports `Client`; the comment in chat_models.py explaining "
        "why we import OpenAI instead is out of date"
    )


def test_the_client_the_wrapper_imports_exists():
    from jarvisclaw import OpenAI  # noqa: PLC0415

    assert OpenAI is not None


def test_constructor_takes_network_not_chain():
    from jarvisclaw import OpenAI  # noqa: PLC0415

    params = inspect.signature(OpenAI.__init__).parameters
    assert "network" in params, "the SDK constructor no longer takes `network`"
    assert "chain" not in params, (
        "the SDK now takes `chain`; the old code passed exactly that and was wrong at "
        "the time — recheck which spelling is current"
    )
    assert "private_key" in params


def test_wallet_client_constructs_with_an_evm_key():
    """The old code could not get this far — it raised ImportError first."""
    from jarvisclaw import OpenAI  # noqa: PLC0415

    client = OpenAI(private_key=DUMMY_EVM_KEY, network="base")
    assert client is not None


def test_the_call_surface_the_wrapper_uses_is_present():
    """_generate calls jc.chat.completions.create(model=…, messages=…, stream=…).

    Asserted attribute by attribute: a missing link in that chain would otherwise
    surface as an AttributeError at request time, on the paid path, in a user's app.
    """
    from jarvisclaw import OpenAI  # noqa: PLC0415

    client = OpenAI(private_key=DUMMY_EVM_KEY, network="base")
    assert hasattr(client, "chat")
    assert hasattr(client.chat, "completions")
    create = client.chat.completions.create
    assert callable(create)

    params = inspect.signature(create).parameters
    for name in ("model", "messages", "stream"):
        assert name in params, f"chat.completions.create no longer accepts {name!r}"


def test_generate_reaches_the_sdk_instead_of_raising_importerror():
    """End-to-end over the branch, with the network call itself stubbed out.

    The guard that matters: _generate must get far enough to build an SDK client and
    invoke it. A sentinel response stands in for the HTTP round trip, so this stays an
    offline test while still proving the ImportError is gone.
    """
    from langchain_core.messages import HumanMessage  # noqa: PLC0415

    from langchain_jarvisclaw import ChatJarvisClaw  # noqa: PLC0415

    import jarvisclaw as sdk  # noqa: PLC0415

    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)

            class Msg:
                content = "stubbed reply"

            class Choice:
                message = Msg()

            class Resp:
                choices = [Choice()]

            return Resp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.chat = FakeChat()

    original = sdk.OpenAI
    sdk.OpenAI = FakeClient
    try:
        chat = ChatJarvisClaw(wallet_private_key=DUMMY_EVM_KEY, model="auto")
        result = chat._generate([HumanMessage(content="hi")])
    finally:
        sdk.OpenAI = original

    assert result.generations[0].message.content == "stubbed reply"
    # Two entries: the constructor kwargs, then the create() kwargs.
    assert len(calls) == 2, f"expected construct + create, got {calls}"
    assert calls[0]["private_key"] == DUMMY_EVM_KEY
    assert calls[0]["network"] == "base"
    assert calls[1]["model"] == "auto"
    assert calls[1]["messages"] == [{"role": "user", "content": "hi"}]


def test_api_key_mode_does_not_touch_the_wallet_path():
    """The branch that always worked must keep working.

    Without wallet_private_key, _generate delegates to ChatOpenAI and must not
    construct an SDK client at all.
    """
    from langchain_core.messages import HumanMessage  # noqa: PLC0415

    from langchain_jarvisclaw import ChatJarvisClaw  # noqa: PLC0415

    import jarvisclaw as sdk  # noqa: PLC0415

    constructed = []

    class ShouldNotBeUsed:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

    original = sdk.OpenAI
    sdk.OpenAI = ShouldNotBeUsed
    try:
        chat = ChatJarvisClaw(api_key="sk-test", model="auto")
        with pytest.raises(Exception):
            # Reaches the real ChatOpenAI and fails on the network / bad key. The
            # assertion is what did NOT happen, checked below.
            chat._generate([HumanMessage(content="hi")])
    finally:
        sdk.OpenAI = original

    assert constructed == [], (
        "API-key mode built a jarvisclaw wallet client; the two paths have been crossed"
    )
