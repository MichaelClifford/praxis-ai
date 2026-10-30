"""Provider-independent session correlation tests (no MLflow server needed)."""

import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

from session_resolver import (
    ASSISTANT_DIGEST_TAG,
    CLIENT_KEY_DIGEST_TAG,
    HISTORY_DIGEST_TAG,
    HISTORY_LENGTH_TAG,
    RESPONSE_ID_TAG,
    SessionResolver,
    USER_DIGEST_TAG,
    USER_LENGTH_TAG,
)


def chat_request(text, key=None):
    request = {"model": "model-a", "messages": [{"role": "user", "content": text}]}
    if key:
        request["prompt_cache_key"] = key
    return request


def chat_response(text, response_id=None):
    response = {"choices": [{"message": {"role": "assistant", "content": text}}]}
    if response_id:
        response["id"] = response_id
    return response


class SessionResolverTests(unittest.TestCase):
    def test_identical_first_prompts_get_unique_random_ids(self):
        resolver = SessionResolver()
        first, _, tags = resolver.resolve(chat_request("hello"), chat_response("hi"))
        second, _, _ = resolver.resolve(chat_request("hello"), chat_response("hi"))
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("praxis-"))
        self.assertNotIn("hello", repr(tags))

    def test_followup_crosses_model_and_provider_without_client_key(self):
        resolver = SessionResolver()
        first, source, _ = resolver.resolve(chat_request("hello"), chat_response("Hello back"))
        self.assertEqual(source, "new")
        followup = {"model": "claude", "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello back"}]},
            {"role": "user", "content": "Follow-up"},
        ]}
        matched, source, _ = resolver.resolve(followup, {"role": "assistant", "content": "done"})
        self.assertEqual((matched, source), (first, "history"))

    def test_followup_without_key_reuses_explicit_session(self):
        resolver = SessionResolver()
        first, _, _ = resolver.resolve(chat_request("hello", "client-session"), chat_response("Hello back"))
        followup = {"messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hello back"},
            {"role": "user", "content": "next"},
        ]}
        matched, source, _ = resolver.resolve(followup, chat_response("done"))
        self.assertEqual((matched, source), (first, "history"))

    def test_provider_switch_matches_two_prior_user_turns_when_assistant_shape_changes(self):
        resolver = SessionResolver()
        resolver.resolve(chat_request("first"), chat_response("answer one"))
        second = {"messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer one"},
            {"role": "user", "content": "second"},
        ]}
        session, _, tags = resolver.resolve(second, chat_response("answer two"))
        self.assertEqual(tags[USER_LENGTH_TAG], "2")
        self.assertEqual(len(tags[USER_DIGEST_TAG]), 64)
        anthropic = {"model": "claude", "messages": [
            {"role": "user", "content": [{"type": "text", "text": "first"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "reformatted answer one"}]},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "answer two"}]},
            {"role": "user", "content": "third"},
        ]}
        matched, source, _ = resolver.resolve(anthropic, {"role": "assistant", "content": "three"})
        self.assertEqual((matched, source), (session, "user_history"))

    def test_same_user_history_in_two_sessions_is_ambiguous(self):
        resolver = SessionResolver()
        for key in ("session-a", "session-b"):
            resolver.resolve({"prompt_cache_key": key, "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "response " + key},
                {"role": "user", "content": "second"},
            ]}, chat_response("done " + key))
        followup = {"messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "provider-specific rendering"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "provider-specific rendering"},
            {"role": "user", "content": "third"},
        ]}
        generated, source, _ = resolver.resolve(followup, chat_response("response"))
        self.assertEqual(source, "new")
        self.assertNotIn(generated, ("session-a", "session-b"))

    def test_switch_to_provider_with_new_client_key_keeps_existing_session(self):
        resolver = SessionResolver()
        first, _, _ = resolver.resolve(chat_request("hello"), chat_response("Hello back"))
        followup = {"prompt_cache_key": "new-provider-key", "input": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hello back"},
            {"role": "user", "content": "Follow-up"},
        ]}
        matched, source, _ = resolver.resolve(followup, {"output": []})
        self.assertEqual((matched, source), (first, "history"))
        delta_only = {"prompt_cache_key": "new-provider-key", "input": "one more turn"}
        same, source, _ = resolver.resolve(delta_only, {"output": []})
        self.assertEqual((same, source), (first, "client_key"))

    def test_different_assistant_replies_disambiguate_identical_openers(self):
        resolver = SessionResolver()
        a, _, _ = resolver.resolve(chat_request("same"), chat_response("answer A"))
        b, _, _ = resolver.resolve(chat_request("same"), chat_response("answer B"))
        self.assertNotEqual(a, b)
        followup = {"messages": [
            {"role": "user", "content": "same"},
            {"role": "assistant", "content": "answer B"},
            {"role": "user", "content": "next"},
        ]}
        matched, source, _ = resolver.resolve(followup, chat_response("more"))
        self.assertEqual((matched, source), (b, "history"))

    def test_ambiguous_history_does_not_merge(self):
        resolver = SessionResolver()
        a, _, _ = resolver.resolve(chat_request("same"), chat_response("same reply"))
        b, _, _ = resolver.resolve(chat_request("same"), chat_response("same reply"))
        continuation = {"messages": [
            {"role": "user", "content": "same"},
            {"role": "assistant", "content": "same reply"},
            {"role": "user", "content": "next"},
        ]}
        third, source, _ = resolver.resolve(continuation, chat_response("answer"))
        self.assertEqual(source, "new")
        self.assertNotIn(third, (a, b))

    def test_matching_prompt_without_assistant_evidence_does_not_merge(self):
        resolver = SessionResolver()
        first, _, _ = resolver.resolve(chat_request("same"), {"choices": []})
        continuation = {"messages": [
            {"role": "user", "content": "same"},
            {"role": "user", "content": "another message"},
        ]}
        second, source, _ = resolver.resolve(continuation, chat_response("answer"))
        self.assertNotEqual(first, second)
        self.assertEqual(source, "new")

    def test_previous_response_id_matches_delta_only_requests(self):
        resolver = SessionResolver()
        first, _, tags = resolver.resolve(
            {"input": "hello"}, {"id": "resp_one", "output": [{"role": "assistant", "content": "hi"}]}
        )
        self.assertEqual(tags[RESPONSE_ID_TAG], "resp_one")
        followup = {"input": [{"role": "user", "content": "next"}], "previous_response_id": "resp_one"}
        matched, source, _ = resolver.resolve(followup, {"id": "resp_two", "output": []})
        self.assertEqual((matched, source), (first, "previous_response_id"))

    def test_truncated_or_unparseable_requests_still_get_unique_ids(self):
        resolver = SessionResolver()
        a, _, _ = resolver.resolve(None, "raw response")
        b, _, _ = resolver.resolve(None, "raw response")
        self.assertNotEqual(a, b)

    def test_restores_fingerprints_from_recent_mlflow_traces(self):
        resolver = SessionResolver()
        first, _, tags = resolver.resolve(chat_request("hello"), chat_response("hi"))
        info = SimpleNamespace(trace_metadata={"mlflow.trace.session": first}, tags=tags,
                               request_time=int(time.time() * 1000))
        client = Mock()
        client.search_traces.return_value = [SimpleNamespace(info=info)]
        restarted = SessionResolver()
        restarted.restore(client, "experiment-1")
        client.search_traces.assert_called_once_with(
            locations=["experiment-1"], max_results=256, include_spans=False,
        )
        continuation = {"messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "again"},
        ]}
        matched, source, _ = restarted.resolve(continuation, chat_response("next"))
        self.assertEqual((matched, source), (first, "history"))

    def test_restore_preserves_new_provider_key_alias(self):
        resolver = SessionResolver()
        original, _, _ = resolver.resolve(chat_request("hello"), chat_response("hi"))
        continuation = {"prompt_cache_key": "provider-key", "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "again"},
        ]}
        matched, _, tags = resolver.resolve(continuation, chat_response("next"))
        self.assertEqual(matched, original)
        self.assertIn(CLIENT_KEY_DIGEST_TAG, tags)
        info = SimpleNamespace(trace_metadata={"mlflow.trace.session": original}, tags=tags,
                               request_time=int(time.time() * 1000))
        client = Mock()
        client.search_traces.return_value = [SimpleNamespace(info=info)]
        restarted = SessionResolver()
        restarted.restore(client, "experiment-1")
        same, source, _ = restarted.resolve({"prompt_cache_key": "provider-key", "input": "new turn"}, {})
        self.assertEqual((same, source), (original, "client_key"))

    def test_restore_keeps_cross_provider_user_history(self):
        resolver = SessionResolver()
        first_request = {"messages": [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "old model's answer"},
            {"role": "user", "content": "two"},
        ]}
        session, _, tags = resolver.resolve(first_request, chat_response("answer two"))
        info = SimpleNamespace(trace_metadata={"mlflow.trace.session": session}, tags=tags,
                               request_time=int(time.time() * 1000))
        client = Mock()
        client.search_traces.return_value = [SimpleNamespace(info=info)]
        restarted = SessionResolver()
        restarted.restore(client, "experiment-1")
        switched = {"messages": [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "formatted differently"},
            {"role": "user", "content": "two"},
            {"role": "assistant", "content": "answer two"},
            {"role": "user", "content": "three"},
        ]}
        matched, source, _ = restarted.resolve(switched, chat_response("answer three"))
        self.assertEqual((matched, source), (session, "user_history"))

    def test_restore_ignores_old_and_expired_traces(self):
        resolver = SessionResolver()
        tags = {HISTORY_LENGTH_TAG: "1", HISTORY_DIGEST_TAG: "a" * 64,
                ASSISTANT_DIGEST_TAG: "b" * 64}
        old = SimpleNamespace(info=SimpleNamespace(trace_metadata={"mlflow.trace.session": "old"},
                                                  tags=tags, request_time=1000))
        legacy = SimpleNamespace(info=SimpleNamespace(trace_metadata={"mlflow.trace.session": "legacy"},
                                                     tags={}, request_time=int(time.time() * 1000)))
        client = Mock()
        client.search_traces.return_value = [legacy, old]
        resolver.restore(client, "experiment-1")
        self.assertFalse(resolver._sessions)

    def test_concurrent_new_requests_do_not_share_ids(self):
        resolver = SessionResolver()
        with ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(pool.map(lambda _: resolver.resolve(chat_request("same"), chat_response("same"))[0],
                                range(32)))
        self.assertEqual(len(ids), len(set(ids)))

    def test_bounded_state_and_expiry(self):
        resolver = SessionResolver(max_sessions=2, max_age=1)
        for text in ("first", "second", "third"):
            resolver.resolve(chat_request(text), chat_response("reply"))
        self.assertEqual(len(resolver._sessions), 2)
        with patch("session_resolver.time.time", return_value=time.time() + 2):
            resolver.resolve(chat_request("fourth"), chat_response("reply"))
        self.assertEqual(len(resolver._sessions), 1)


if __name__ == "__main__":
    unittest.main()
