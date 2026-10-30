"""Best-effort, provider-independent session correlation for mirrored LLM calls.

Only digests and opaque response IDs are retained. A prompt alone never joins
an existing session: two unrelated conversations can start identically.
"""

import hashlib
import json
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass


HISTORY_LENGTH_TAG = "praxis.session.history_length"
HISTORY_DIGEST_TAG = "praxis.session.history_digest"
USER_LENGTH_TAG = "praxis.session.user_length"
USER_DIGEST_TAG = "praxis.session.user_digest"
ASSISTANT_DIGEST_TAG = "praxis.session.assistant_digest"
RESPONSE_ID_TAG = "praxis.session.response_id"
CLIENT_KEY_DIGEST_TAG = "praxis.session.client_key_digest"
SESSION_SOURCE_TAG = "praxis.session.source"
MAX_SESSIONS = 256
MAX_AGE_SECONDS = 24 * 60 * 60


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            if part.get("type") in ("text", "input_text", "output_text", "text_delta"):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
            elif part.get("type") == "tool_result":
                result = _text(part.get("content"))
                if result is not None:
                    parts.append(result)
    return "".join(parts) if parts else None


def _message(item):
    if not isinstance(item, dict):
        return None
    role = item.get("role")
    if item.get("type") == "function_call_output":
        role, content = "tool", item.get("output")
    elif role in ("user", "assistant", "tool"):
        content = item.get("content")
    else:
        return None
    text = _text(content)
    if not text or not text.strip():
        return None
    return _digest((role, text))


def _request_items(request):
    if not isinstance(request, dict):
        return []
    items = request.get("messages")
    if not isinstance(items, list):
        items = request.get("input")
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    return items if isinstance(items, list) else []


def history_digests(request):
    """Hash semantic messages, ignoring model, protocol and cache controls."""
    return [digest for item in _request_items(request) if (digest := _message(item))]


def user_digests(request):
    """Hash only user-authored text, excluding tool results and provider metadata."""
    result = []
    for item in _request_items(request):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, list):
            content = [part for part in content if isinstance(part, dict)
                       and part.get("type") in ("text", "input_text", "output_text")]
        text = _text(content)
        if text and text.strip():
            result.append(_digest(("user", text)))
    return result


def history_prefixes(messages):
    """Create a prefix digest per message for constant-size state storage."""
    running = hashlib.sha256()
    prefixes = []
    for digest in messages:
        running.update(bytes.fromhex(digest))
        prefixes.append(running.hexdigest())
    return prefixes


def assistant_digest(response, raw_response=""):
    """Hash final assistant text to disambiguate identical opening prompts."""
    if isinstance(response, dict):
        if isinstance(response.get("choices"), list):
            for choice in response["choices"]:
                if isinstance(choice, dict) and (digest := _message(choice.get("message"))):
                    return digest
        if isinstance(response.get("output"), list):
            for item in response["output"]:
                if isinstance(item, dict) and (digest := _message(item)):
                    return digest
        if response.get("role") == "assistant" and (digest := _message(response)):
            return digest
        if isinstance(response.get("content"), list):
            return _message({"role": "assistant", "content": response["content"]})
    # Anthropic SSE is not currently reconstructed by the display adapter.
    parts = []
    for line in raw_response.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[5:].strip())
        except (ValueError, TypeError):
            continue
        if isinstance(event, dict) and event.get("type") == "content_block_delta":
            delta = event.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                parts.append(delta["text"])
    return _message({"role": "assistant", "content": "".join(parts)}) if parts else None


def response_id(response):
    if isinstance(response, dict) and isinstance(response.get("id"), str) and len(response["id"]) <= 256:
        return response["id"]
    return None


@dataclass
class _State:
    session_id: str
    history_length: int
    history_digest: str
    user_length: int
    user_digest: str
    assistant_digest: str | None
    response_id: str | None
    last_seen: float


class SessionResolver:
    """Bounded and thread-safe correlation of successive HTTP requests."""

    def __init__(self, max_sessions=MAX_SESSIONS, max_age=MAX_AGE_SECONDS):
        self._sessions = OrderedDict()
        self._aliases = OrderedDict()
        self._lock = threading.Lock()
        self._max_sessions = max_sessions
        self._max_age = max_age

    def restore(self, client, experiment_id):
        """Best-effort warmup from recently persisted MLflow trace metadata."""
        traces = client.search_traces(locations=[experiment_id], max_results=self._max_sessions,
                                      include_spans=False)
        now = time.time()
        with self._lock:
            for trace in sorted(traces, key=lambda item: item.info.request_time):
                info = trace.info
                session = info.trace_metadata.get("mlflow.trace.session")
                tags = info.tags
                try:
                    length = int(tags[HISTORY_LENGTH_TAG])
                    digest = tags[HISTORY_DIGEST_TAG]
                    user_length = int(tags.get(USER_LENGTH_TAG, 0))
                except (KeyError, TypeError, ValueError):
                    continue  # Old adapter traces have no correlation fingerprints.
                if not session or length < 1 or len(digest) != 64:
                    continue
                user_digest = tags.get(USER_DIGEST_TAG, "")
                seen = info.request_time / 1000
                if now - seen > self._max_age:
                    continue
                self._remember(_State(session, length, digest, user_length, user_digest,
                                      tags.get(ASSISTANT_DIGEST_TAG), tags.get(RESPONSE_ID_TAG), seen))
                if client_key_digest := tags.get(CLIENT_KEY_DIGEST_TAG):
                    self._aliases[client_key_digest] = session

    def resolve(self, request, response, raw_response=""):
        """Return (session ID, correlation source, tags) for one mirrored call."""
        now = time.time()
        messages = history_digests(request)
        prefixes = history_prefixes(messages)
        users = user_digests(request)
        user_prefixes = history_prefixes(users)
        explicit = request.get("prompt_cache_key") if isinstance(request, dict) else None
        explicit = explicit if isinstance(explicit, str) and explicit.strip() else None
        key_digest = hashlib.sha256(explicit.encode("utf-8")).hexdigest() if explicit else None
        previous = request.get("previous_response_id") if isinstance(request, dict) else None

        with self._lock:
            self._expire(now)
            match, source = self._match(explicit, key_digest, previous, messages, prefixes, users, user_prefixes)
            session = match.session_id if match else explicit or f"praxis-{secrets.token_urlsafe(18)}"
            source = source if match else "client_key" if explicit else "new"
            tags = {SESSION_SOURCE_TAG: source}
            if key_digest:
                tags[CLIENT_KEY_DIGEST_TAG] = key_digest
                self._aliases[key_digest] = session
                self._aliases.move_to_end(key_digest)
                while len(self._aliases) > 2 * self._max_sessions:
                    self._aliases.popitem(last=False)
            if prefixes:
                tags[HISTORY_LENGTH_TAG] = str(len(prefixes))
                tags[HISTORY_DIGEST_TAG] = prefixes[-1]
            if user_prefixes:
                tags[USER_LENGTH_TAG] = str(len(user_prefixes))
                tags[USER_DIGEST_TAG] = user_prefixes[-1]
            assistant = assistant_digest(response, raw_response)
            if assistant:
                tags[ASSISTANT_DIGEST_TAG] = assistant
            completed_id = response_id(response)
            if completed_id:
                tags[RESPONSE_ID_TAG] = completed_id
            if prefixes or completed_id:
                self._remember(_State(session, len(prefixes), prefixes[-1] if prefixes else "",
                                      len(user_prefixes), user_prefixes[-1] if user_prefixes else "",
                                      assistant, completed_id, now))
            return session, source, tags

    def _match(self, explicit, key_digest, previous, messages, prefixes, users, user_prefixes):
        if explicit:
            if keyed := self._sessions.get(self._aliases.get(key_digest, explicit)):
                return keyed, "client_key"
        if isinstance(previous, str) and previous:
            linked = {state.session_id: state for state in self._sessions.values()
                      if state.response_id == previous}
            if len(linked) == 1:
                return next(iter(linked.values())), "previous_response_id"
        # A single message cannot distinguish a continuation from another
        # conversation starting with the same user prompt.
        if len(messages) < 2:
            return None, None
        candidates = []
        for state in self._sessions.values():
            length = state.history_length
            if length < 1 or length >= len(messages) or state.history_digest != prefixes[length - 1]:
                continue
            if length == 1 and not state.assistant_digest:
                continue  # A matching first prompt alone is not enough evidence.
            if state.assistant_digest and messages[length] != state.assistant_digest:
                continue
            candidates.append(state)
        if candidates:
            longest = max(state.history_length for state in candidates)
            matches = {state.session_id: state for state in candidates if state.history_length == longest}
            if len(matches) == 1:
                return next(iter(matches.values())), "history"
            return None, None
        # Provider switches can reformat assistant/tool messages. Two prior
        # matching user turns provide a weaker but useful independent signal;
        # one matching opening prompt never does.
        if len(users) < 3:
            return None, None
        candidates = [state for state in self._sessions.values()
                      if 2 <= state.user_length < len(users)
                      and state.user_digest == user_prefixes[state.user_length - 1]]
        if candidates:
            longest = max(state.user_length for state in candidates)
            matches = {state.session_id: state for state in candidates if state.user_length == longest}
            if len(matches) == 1:
                return next(iter(matches.values())), "user_history"
        return None, None

    def _remember(self, state):
        self._sessions[state.session_id] = state
        self._sessions.move_to_end(state.session_id)
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)

    def _expire(self, now):
        for key, state in list(self._sessions.items()):
            if now - state.last_seen > self._max_age:
                del self._sessions[key]
        for key, session in list(self._aliases.items()):
            if session not in self._sessions:
                del self._aliases[key]
