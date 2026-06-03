"""
Provider-neutral chat-request types.

This module is the keystone of the drop-in compatibility layer. Every
public entry-point (``ReliabilityModule.run``, ``agentcodec.openai.OpenAI``,
``agentcodec.anthropic.Anthropic``, ``agentcodec.ollama.Client``) converts
its native input shape into a :class:`ChatRequest`. Every internal LLM
call sees the same ``ChatRequest``. Each transport (`_transmit_openai`,
`_transmit_anthropic`, `_transmit_ollama_*`) converts back into the
provider's native payload at the last possible moment.

The design rule is: **`ChatRequest` is the lingua franca**. Anything
provider-specific (Anthropic content blocks, Ollama's flat message
list, OpenAI's tool_call schema) lives in the per-provider ``from_*`` /
``to_*`` helpers, never on the call sites in between.

Frozen dataclasses are used throughout because techniques freely pass
requests to multiple concurrent channels; an accidental in-place mutation
on one branch would corrupt the others. The ``with_*`` helpers preserve
the immutable contract by returning fresh instances.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Content blocks — used inside Message.content for multimodal / tool flows
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextBlock:
    """Plain text segment. The common case; ``Message(content="hi")`` auto-wraps."""
    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True, slots=True)
class ImageBlock:
    """Multimodal image input.

    Carries a superset of provider shapes; the per-provider serializer
    extracts the fields it needs. Set ``url`` for OpenAI-compat (HTTPS or
    ``data:`` URI), or ``data`` + ``media_type`` for Anthropic-style base64.
    """
    url: str | None = None
    data: str | None = None              # base64-encoded (Anthropic style)
    media_type: str | None = None        # "image/png" | "image/jpeg" | ...
    detail: Literal["auto", "low", "high"] | None = None  # OpenAI knob
    type: Literal["image"] = "image"


@dataclass(frozen=True, slots=True)
class ToolUseBlock:
    """Assistant turn requesting a tool call.

    Anthropic surfaces these as content blocks inside an assistant message.
    OpenAI uses a parallel ``tool_calls`` list on the message; the
    serializer translates either direction.
    """
    id: str
    name: str
    input: Mapping[str, Any]
    type: Literal["tool_use"] = "tool_use"


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    """Tool-result content block (Anthropic shape).

    OpenAI conveys the same information via a ``role="tool"`` message with
    ``tool_call_id``; the serializer handles both.
    """
    tool_use_id: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


ContentBlock = TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock


# ---------------------------------------------------------------------------
# Tool call — OpenAI's "arguments as JSON string" shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A single tool/function invocation requested by the model.

    Mirrors OpenAI's shape because (a) it's the most common starting
    point, and (b) JSON-string arguments serialize losslessly to every
    provider. The Anthropic serializer expands ``arguments`` into a dict
    before sending.
    """
    id: str
    name: str
    arguments: str   # JSON string, OpenAI convention


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Message:
    """A single conversation turn.

    ``content`` is either a plain string (the 99% case) or a tuple of
    :class:`ContentBlock` instances for multimodal / tool flows. The
    public constructors accept either form; per-provider serializers
    handle the conversion.
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: str | tuple[ContentBlock, ...]
    tool_calls: tuple[ToolCall, ...] | None = None   # assistant turn with tool calls
    tool_call_id: str | None = None                  # tool-role turns reference the call
    name: str | None = None                          # optional function/tool name

    @property
    def text(self) -> str:
        """Flatten content to a plain string. Drops non-text blocks silently."""
        if isinstance(self.content, str):
            return self.content
        return "".join(
            block.text for block in self.content if isinstance(block, TextBlock)
        )

    @property
    def has_non_text_content(self) -> bool:
        """True iff content contains anything other than plain text."""
        if isinstance(self.content, str):
            return False
        return any(not isinstance(b, TextBlock) for b in self.content)


# ---------------------------------------------------------------------------
# ChatRequest — provider-neutral request envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """Provider-neutral chat request. Immutable.

    All public ``mod.run`` / shim ``create`` paths funnel into one of these
    before reaching ``AgentChannel.transmit``. Techniques that need to
    mutate the prompt (HARQ critique, MoA aggregator, voting templates,
    ...) call :meth:`with_user` to derive a new request with the last
    user turn replaced — the system prompt, prior history, and tools are
    carried through unchanged.
    """
    messages: tuple[Message, ...]
    temperature: float | None = None
    max_tokens: int | None = None
    tools: tuple[Mapping[str, Any], ...] | None = None
    tool_choice: str | Mapping[str, Any] | None = None
    response_format: Mapping[str, Any] | None = None
    stop: tuple[str, ...] | None = None
    seed: int | None = None
    top_p: float | None = None
    request_logprobs: bool = False
    extra_body: Mapping[str, Any] | None = None
    # Free-form provider-passthrough kwargs the shim collected and didn't
    # understand. These are dropped by the reliability path but useful for
    # the pure-passthrough debug case.
    extra_kwargs: Mapping[str, Any] | None = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_prompt(
        cls,
        prompt: str,
        *,
        system: str | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        response_format: Mapping[str, Any] | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatRequest:
        """Build a single-turn request from a plain prompt string.

        This is what the back-compat ``transmit("prompt string")`` overload
        calls under the hood, and what users get by default from
        ``mod.run("some prompt")``.
        """
        msgs: list[Message] = []
        if system:
            msgs.append(Message(role="system", content=system))
        msgs.append(Message(role="user", content=prompt))
        return cls(
            messages=tuple(msgs),
            temperature=temperature,
            max_tokens=max_tokens,
            tools=_freeze_tuple(tools),
            tool_choice=tool_choice,
            response_format=response_format,
            stop=_freeze_stop(stop),
            seed=seed,
            top_p=top_p,
        )

    @classmethod
    def from_openai_messages(
        cls,
        messages: Sequence[Mapping[str, Any] | Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        response_format: Mapping[str, Any] | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_kwargs: Mapping[str, Any] | None = None,
    ) -> ChatRequest:
        """Build from OpenAI ``messages=[...]`` payload (also accepts ``Message`` instances).

        This is the constructor the OpenAI compat shim uses on every
        ``client.chat.completions.create(messages=[...])`` call.
        """
        msgs = tuple(_coerce_openai_message(m) for m in messages)
        return cls(
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=_freeze_tuple(tools),
            tool_choice=tool_choice,
            response_format=response_format,
            stop=_freeze_stop(stop),
            seed=seed,
            top_p=top_p,
            extra_body=extra_body,
            extra_kwargs=extra_kwargs,
        )

    @classmethod
    def from_anthropic(
        cls,
        *,
        messages: Sequence[Mapping[str, Any] | Message],
        system: str | Sequence[Mapping[str, Any]] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        stop_sequences: str | Sequence[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        extra_kwargs: Mapping[str, Any] | None = None,
    ) -> ChatRequest:
        """Build from an Anthropic ``messages.create(...)`` payload.

        Anthropic carries ``system`` as a top-level kwarg (not a message);
        we hoist it into a leading ``system`` message so the rest of the
        pipeline sees a uniform shape. Anthropic's ``tool_choice`` and
        ``stop_sequences`` map directly to our ``tool_choice`` / ``stop``.
        """
        msgs: list[Message] = []
        if system:
            system_text = _flatten_anthropic_system(system)
            if system_text:
                msgs.append(Message(role="system", content=system_text))
        msgs.extend(_coerce_anthropic_message(m) for m in messages)
        return cls(
            messages=tuple(msgs),
            temperature=temperature,
            max_tokens=max_tokens,
            tools=_freeze_tuple(_translate_anthropic_tools(tools)) if tools else None,
            tool_choice=tool_choice,
            stop=_freeze_stop(stop_sequences),
            top_p=top_p,
            extra_kwargs=extra_kwargs,
        )

    @classmethod
    def from_ollama_messages(
        cls,
        messages: Sequence[Mapping[str, Any] | Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        options: Mapping[str, Any] | None = None,
        format: str | Mapping[str, Any] | None = None,
        extra_kwargs: Mapping[str, Any] | None = None,
    ) -> ChatRequest:
        """Build from ``ollama.chat(model, messages, ...)`` payload.

        Ollama's ``options`` dict carries temperature / seed / etc. We
        extract the recognized ones; the rest stay in ``extra_body`` so
        the channel can pass them through.
        """
        opts: dict[str, Any] = dict(options or {})
        temperature = opts.pop("temperature", None)
        seed = opts.pop("seed", None)
        top_p = opts.pop("top_p", None)
        stop_raw = opts.pop("stop", None)
        max_tokens = opts.pop("num_predict", None)
        response_format = (
            {"type": "json_object"} if format == "json"
            else format if isinstance(format, Mapping)
            else None
        )
        msgs = tuple(_coerce_openai_message(m) for m in messages)  # ollama shares OpenAI shape
        return cls(
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=_freeze_tuple(tools),
            response_format=response_format,
            stop=_freeze_stop(stop_raw),
            seed=seed,
            top_p=top_p,
            extra_body=opts or None,
            extra_kwargs=extra_kwargs,
        )

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    @property
    def system(self) -> str | None:
        """The leading system message's text, or None."""
        for m in self.messages:
            if m.role == "system":
                return m.text
            break
        return None

    @property
    def history(self) -> tuple[Message, ...]:
        """All messages excluding the final user turn (system + earlier turns).

        Used by techniques as the immutable conversation prefix; the
        technique mutates only the final user turn and then calls
        :meth:`with_user` to derive a new request.
        """
        if not self.messages:
            return ()
        last_user_idx = self._last_user_index()
        if last_user_idx is None:
            return self.messages
        return self.messages[:last_user_idx]

    @property
    def last_user_text(self) -> str:
        """Plain-text content of the final user message.

        This is what techniques get when they read ``task.prompt`` —
        the back-compat string view of the request.
        """
        idx = self._last_user_index()
        if idx is None:
            raise ValueError("ChatRequest has no user message")
        return self.messages[idx].text

    def _last_user_index(self) -> int | None:
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].role == "user":
                return i
        return None

    # ------------------------------------------------------------------
    # Mutation helpers — return new instances
    # ------------------------------------------------------------------

    def with_user(self, content: str | Sequence[ContentBlock]) -> ChatRequest:
        """Return a new ChatRequest with the last user turn's content replaced.

        Techniques that re-prompt (HARQ critic round, MoA aggregator,
        voting templates, etc.) call this to preserve system, history,
        and tool config while substituting the user-facing text.

        If the request has no user turn yet, appends one.
        """
        new_content: str | tuple[ContentBlock, ...] = (
            content if isinstance(content, str) else tuple(content)
        )
        idx = self._last_user_index()
        if idx is None:
            new_msgs = (*self.messages, Message(role="user", content=new_content))
        else:
            new_msgs = (
                (*self.messages[:idx], Message(role="user", content=new_content), *self.messages[idx + 1:])
            )
        return replace(self, messages=new_msgs)

    def with_system(self, system: str) -> ChatRequest:
        """Return a new ChatRequest with the leading system message set/replaced."""
        new_sys = Message(role="system", content=system)
        if self.messages and self.messages[0].role == "system":
            new_msgs = (new_sys, *self.messages[1:])
        else:
            new_msgs = (new_sys, *self.messages)
        return replace(self, messages=new_msgs)

    def with_appended_assistant(
        self, content: str | Sequence[ContentBlock],
        *,
        tool_calls: Sequence[ToolCall] | None = None,
    ) -> ChatRequest:
        """Append an assistant turn. Used when chaining tool-use cycles."""
        new_content: str | tuple[ContentBlock, ...] = (
            content if isinstance(content, str) else tuple(content)
        )
        msg = Message(
            role="assistant",
            content=new_content,
            tool_calls=tuple(tool_calls) if tool_calls else None,
        )
        return replace(self, messages=(*self.messages, msg))

    def with_extra(self, **overrides: Any) -> ChatRequest:
        """Replace top-level fields. Convenience over ``dataclasses.replace``."""
        return replace(self, **overrides)

    # ------------------------------------------------------------------
    # Provider serializers — used by AgentChannel._transmit_*
    # ------------------------------------------------------------------

    def to_openai_messages(self) -> list[dict[str, Any]]:
        """Serialize to the OpenAI ``messages=[...]`` payload shape."""
        return [_message_to_openai(m) for m in self.messages]

    def to_anthropic_payload(self) -> tuple[str | None, list[dict[str, Any]]]:
        """Serialize to ``(system, messages)`` for ``anthropic.messages.create``."""
        system_text: str | None = None
        msgs: list[dict[str, Any]] = []
        for m in self.messages:
            if m.role == "system":
                # Anthropic system is a top-level kwarg, not a message.
                # Concatenate multiple system messages (rare) with two newlines.
                system_text = m.text if system_text is None else f"{system_text}\n\n{m.text}"
            else:
                msgs.append(_message_to_anthropic(m))
        return system_text, msgs

    def to_ollama_messages(self) -> list[dict[str, Any]]:
        """Serialize to ``ollama.chat(messages=[...])`` payload.

        Ollama follows the OpenAI message shape closely, so we reuse the
        OpenAI serializer and only differ where needed (image attachments
        use ``images=[base64...]`` instead of multimodal content blocks).
        """
        out: list[dict[str, Any]] = []
        for m in self.messages:
            base = _message_to_openai(m)
            # Translate image blocks to ollama's images= array on the message.
            if isinstance(m.content, tuple):
                images = [b for b in m.content if isinstance(b, ImageBlock)]
                if images:
                    base["images"] = [img.data or img.url or "" for img in images]
            out.append(base)
        return out


# ---------------------------------------------------------------------------
# ChatResponse — used by the compat shims to bundle reliability output
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Provider-neutral assistant response.

    Carries the assistant turn plus reliability metadata (technique used,
    cost, latency, trace). Each provider shim wraps this in its native
    response shape (OpenAI's ``ChatCompletion``, Anthropic's ``Message``,
    Ollama's dict).
    """
    text: str
    tool_calls: tuple[ToolCall, ...] | None = None
    finish_reason: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    technique_used: str | None = None
    cost_usd: float = 0.0
    latency_s: float = 0.0
    # Carries the original ReliabilityResult for users who want the full trace.
    reliability_result: Any | None = None


# ---------------------------------------------------------------------------
# Streaming chunks — yielded by AgentChannel.atransmit_stream()
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChannelChunk:
    """A single incremental frame from `AgentChannel.atransmit_stream()`.

    Stream protocol: one or more ``ChannelChunk`` instances followed by
    exactly one terminal :class:`ChannelDone` carrying the aggregated
    :class:`AgentOutput`. Techniques that compose multiple channel calls
    are expected to forward these as their own typed events (see
    :class:`TokenEvent`, role-tagged).

    ``role`` distinguishes the kind of content the delta represents:

    * ``"answer"`` — the final user-facing text the model emitted.
      Concatenating all ``answer`` chunks yields the response text.
    * ``"thinking"`` — model-internal reasoning, captured from the
      provider's separate channel (Anthropic ``ThinkingBlock``, OpenAI
      ``reasoning_content``, Ollama ``msg.thinking``) or inline
      ``<think>...</think>`` tags. Concatenating all ``thinking`` chunks
      yields the captured ``thinking_text``.
    * ``"tool_call"`` — a tool/function invocation. ``tool_call`` is
      set; ``text`` is empty. May be emitted multiple times for the same
      call as the provider streams partial JSON arguments.

    Do NOT use this for technique-level deliberation (drafts, critiques,
    candidates, syntheses). Those are :class:`TokenEvent`s with the
    appropriate role, emitted by the technique layer above the channel.
    """
    role: Literal["answer", "thinking", "tool_call"]
    text: str = ""
    tool_call: ToolCall | None = None


@dataclass(frozen=True, slots=True)
class ChannelDone:
    """Terminal frame from `AgentChannel.atransmit_stream()`.

    Carries the fully-aggregated :class:`AgentOutput` — same shape as the
    sync :meth:`AgentChannel.transmit` would have returned. Techniques
    that compose channel calls should use this to drive their own
    accumulation, cost tracking, and progress events.
    """
    output: Any   # AgentOutput — typed as Any to avoid a models.py cycle


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _freeze_tuple(items: Iterable[Any] | None) -> tuple[Any, ...] | None:
    if items is None:
        return None
    out = tuple(items)
    return out or None


def _freeze_stop(stop: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if stop is None:
        return None
    if isinstance(stop, str):
        return (stop,)
    out = tuple(stop)
    return out or None


def _coerce_openai_message(m: Mapping[str, Any] | Message) -> Message:
    """Translate an OpenAI-shaped dict (or pass-through a Message) to Message."""
    if isinstance(m, Message):
        return m
    role = m.get("role")
    if role not in ("system", "user", "assistant", "tool"):
        raise ValueError(f"unknown message role: {role!r}")
    content = m.get("content")
    blocks = _coerce_openai_content(content)
    tool_calls = m.get("tool_calls")
    return Message(
        role=role,
        content=blocks,
        tool_calls=(
            tuple(_coerce_tool_call(tc) for tc in tool_calls)
            if tool_calls else None
        ),
        tool_call_id=m.get("tool_call_id"),
        name=m.get("name"),
    )


def _coerce_openai_content(
    content: str | None | Sequence[Mapping[str, Any]],
) -> str | tuple[ContentBlock, ...]:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    blocks: list[ContentBlock] = []
    for raw in content:
        t = raw.get("type")
        if t == "text":
            blocks.append(TextBlock(text=raw.get("text", "")))
        elif t == "image_url":
            url_field = raw.get("image_url") or {}
            url = url_field.get("url") if isinstance(url_field, Mapping) else url_field
            blocks.append(ImageBlock(url=url, detail=url_field.get("detail") if isinstance(url_field, Mapping) else None))
        elif t == "image":
            # Anthropic-style on the way in (rare via OpenAI shim, but tolerate)
            src = raw.get("source") or {}
            blocks.append(ImageBlock(
                data=src.get("data"),
                media_type=src.get("media_type"),
            ))
        else:
            # Unknown block type — preserve as text-flatten fallback.
            blocks.append(TextBlock(text=str(raw)))
    return tuple(blocks)


def _coerce_tool_call(tc: Mapping[str, Any]) -> ToolCall:
    fn = tc.get("function") or {}
    return ToolCall(
        id=tc.get("id", ""),
        name=fn.get("name") or tc.get("name", ""),
        arguments=fn.get("arguments") if isinstance(fn.get("arguments"), str) else (
            __import__("json").dumps(fn.get("arguments") or {})
        ),
    )


def _coerce_anthropic_message(m: Mapping[str, Any] | Message) -> Message:
    if isinstance(m, Message):
        return m
    role = m.get("role")
    if role not in ("user", "assistant"):
        raise ValueError(f"Anthropic messages must have role user|assistant, got {role!r}")
    content = m.get("content")
    blocks = _coerce_anthropic_content(content)
    return Message(role=role, content=blocks)


def _coerce_anthropic_content(
    content: str | Sequence[Mapping[str, Any]] | None,
) -> str | tuple[ContentBlock, ...]:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    blocks: list[ContentBlock] = []
    for raw in content:
        t = raw.get("type")
        if t == "text":
            blocks.append(TextBlock(text=raw.get("text", "")))
        elif t == "image":
            src = raw.get("source") or {}
            blocks.append(ImageBlock(
                data=src.get("data"),
                media_type=src.get("media_type"),
                url=src.get("url"),
            ))
        elif t == "tool_use":
            blocks.append(ToolUseBlock(
                id=raw.get("id", ""),
                name=raw.get("name", ""),
                input=raw.get("input") or {},
            ))
        elif t == "tool_result":
            res_content = raw.get("content")
            if isinstance(res_content, list):
                # Anthropic tool_result content can be a list of blocks too; flatten.
                res_content = "".join(
                    b.get("text", "") for b in res_content
                    if isinstance(b, Mapping) and b.get("type") == "text"
                )
            blocks.append(ToolResultBlock(
                tool_use_id=raw.get("tool_use_id", ""),
                content=res_content if isinstance(res_content, str) else str(res_content),
                is_error=bool(raw.get("is_error")),
            ))
        else:
            blocks.append(TextBlock(text=str(raw)))
    return tuple(blocks)


def _flatten_anthropic_system(
    system: str | Sequence[Mapping[str, Any]],
) -> str:
    if isinstance(system, str):
        return system
    parts = []
    for entry in system:
        if isinstance(entry, Mapping) and entry.get("type") == "text":
            parts.append(entry.get("text", ""))
    return "\n\n".join(p for p in parts if p)


def _translate_anthropic_tools(
    tools: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Translate Anthropic tool definitions to OpenAI tool shape.

    Anthropic: ``{"name": "...", "description": "...", "input_schema": {...}}``
    OpenAI:    ``{"type": "function", "function": {"name", "description", "parameters"}}``

    We canonicalize on OpenAI's shape internally; the Anthropic transport
    serializer translates back to Anthropic's shape when sending.
    """
    if not tools:
        return None
    out = []
    for t in tools:
        if "function" in t:
            out.append(dict(t))
        else:
            out.append({
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description"),
                    "parameters": t.get("input_schema") or {},
                },
            })
    return out


# --- per-message serializers ------------------------------------------------


def _message_to_openai(m: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role}
    if isinstance(m.content, str):
        out["content"] = m.content
    else:
        # Multi-block content. Translate to OpenAI's content-block array
        # for messages that have images; otherwise flatten to a string for
        # max provider compatibility.
        if any(not isinstance(b, TextBlock) for b in m.content):
            blocks_out: list[dict[str, Any]] = []
            for b in m.content:
                if isinstance(b, TextBlock):
                    blocks_out.append({"type": "text", "text": b.text})
                elif isinstance(b, ImageBlock):
                    url = b.url or (
                        f"data:{b.media_type or 'image/png'};base64,{b.data}"
                        if b.data else ""
                    )
                    image_url: dict[str, Any] = {"url": url}
                    if b.detail:
                        image_url["detail"] = b.detail
                    blocks_out.append({"type": "image_url", "image_url": image_url})
                elif isinstance(b, ToolUseBlock):
                    # OpenAI puts tool_use on the message's tool_calls list, not in content
                    pass
                elif isinstance(b, ToolResultBlock):
                    blocks_out.append({"type": "text", "text": b.content})
            out["content"] = blocks_out or ""
        else:
            out["content"] = "".join(b.text for b in m.content if isinstance(b, TextBlock))
    if m.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in m.tool_calls
        ]
    if m.tool_call_id:
        out["tool_call_id"] = m.tool_call_id
    if m.name:
        out["name"] = m.name
    return out


def _message_to_anthropic(m: Message) -> dict[str, Any]:
    """Serialize a non-system message to Anthropic's shape.

    System messages are hoisted into the top-level ``system`` kwarg by
    :meth:`ChatRequest.to_anthropic_payload`, so this never sees them.
    """
    if m.role == "tool":
        # Anthropic represents tool results as a user message with
        # tool_result content blocks.
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": m.tool_call_id or "",
                "content": m.text,
            }],
        }
    # Build content blocks.
    if isinstance(m.content, str):
        if m.tool_calls:
            content_blocks: list[dict[str, Any]] = []
            if m.content:
                content_blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                import json as _json
                try:
                    tc_input = _json.loads(tc.arguments) if tc.arguments else {}
                except _json.JSONDecodeError:
                    tc_input = {"_raw": tc.arguments}
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc_input,
                })
            return {"role": m.role, "content": content_blocks}
        return {"role": m.role, "content": m.content}
    # Multi-block.
    blocks_out: list[dict[str, Any]] = []
    for b in m.content:
        if isinstance(b, TextBlock):
            blocks_out.append({"type": "text", "text": b.text})
        elif isinstance(b, ImageBlock):
            if b.data:
                blocks_out.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": b.media_type or "image/png",
                        "data": b.data,
                    },
                })
            elif b.url:
                blocks_out.append({
                    "type": "image",
                    "source": {"type": "url", "url": b.url},
                })
        elif isinstance(b, ToolUseBlock):
            blocks_out.append({
                "type": "tool_use",
                "id": b.id,
                "name": b.name,
                "input": dict(b.input),
            })
        elif isinstance(b, ToolResultBlock):
            blocks_out.append({
                "type": "tool_result",
                "tool_use_id": b.tool_use_id,
                "content": b.content,
                "is_error": b.is_error,
            })
    if m.tool_calls:
        for tc in m.tool_calls:
            import json as _json
            try:
                tc_input = _json.loads(tc.arguments) if tc.arguments else {}
            except _json.JSONDecodeError:
                tc_input = {"_raw": tc.arguments}
            blocks_out.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc_input,
            })
    return {"role": m.role, "content": blocks_out}


__all__ = [
    "ChannelChunk",
    "ChannelDone",
    "ChatRequest",
    "ChatResponse",
    "ContentBlock",
    "ImageBlock",
    "Message",
    "TextBlock",
    "ToolCall",
    "ToolResultBlock",
    "ToolUseBlock",
]
