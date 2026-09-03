"""Behavioral test for the mem0 shutdown race.

Root cause (see card t_e87714cc): server-side LLM fact extraction inside
``backend.add(infer=True)`` takes 10-90 s, but ``shutdown()`` joined the
mem0-sync thread for only 5 s before calling ``_shutdown_backend()``. The
backend closed the Qdrant client mid-insert, and mem0's insert path only
logged the resulting ``Cannot send a request, as the client has been closed``
error — the memory was silently dropped.

This test drives the REAL provider (``sync_turn`` + ``shutdown``) against a
REAL Qdrant instance on ``:6333``. The only stand-in is the backend double,
which replaces the slow LLM extraction with a bounded ``time.sleep`` and then
upserts a real point into Qdrant — the same client the real backend would
close. Assertions are on actual Qdrant state (the point is retrievable by id
after shutdown) and on whether ``close()`` was called while ``add()`` was
still in flight, not on log strings.
"""

import threading
import time
import uuid

import pytest

from plugins.memory.mem0 import Mem0MemoryProvider

QDRANT_URL = "http://localhost:6333"
TEST_COLLECTION = "mem0_shutdown_race_test"

# Must exceed the OLD shutdown() sync join (5.0 s) so the pre-fix code path
# closes the client mid-add, while staying far under the NEW generous join
# (_SYNC_SHUTDOWN_WAIT_SECS = 120). 7 s gives ~2 s of scheduling margin over
# the 5 s join without dragging the suite down.
EXTRACTION_DELAY_SECS = 7.0


class SlowQdrantBackend:
    """Backend double: block (simulated LLM extraction), then upsert to Qdrant.

    ``add`` never catches the Qdrant exception — if ``close`` runs mid-add,
    the upsert raises ``ResponseHandlingException: Cannot send a request, as
    the client has been closed`` and propagates exactly like the real mem0
    insert path. That failure is what the test asserts against.
    """

    def __init__(self, client, collection_name):
        self.client = client
        self.collection_name = collection_name
        self.add_entered = threading.Event()
        self.add_finished = threading.Event()
        self.closed = threading.Event()
        self._in_add = threading.Event()
        self.closed_during_add = False
        self.last_point_id = None

    def add(self, messages, *, user_id, agent_id, infer=False, metadata=None):
        self._in_add.set()
        self.add_entered.set()
        try:
            time.sleep(EXTRACTION_DELAY_SECS)
            point_id = str(uuid.uuid4())
            self.client.upsert(
                collection_name=self.collection_name,
                points=[{
                    "id": point_id,
                    "vector": [0.0] * 8,
                    "payload": {
                        "user_id": user_id,
                        "agent_id": agent_id,
                        "memory": messages[0]["content"] if messages else "",
                    },
                }],
            )
            self.last_point_id = point_id
            return {"results": [{"id": point_id}]}
        finally:
            self._in_add.clear()
            self.add_finished.set()

    def close(self):
        if self._in_add.is_set():
            self.closed_during_add = True
        self.client.close()
        self.closed.set()

    def search(self, query, *, filters, top_k=10, rerank=False):
        return []

    def update(self, memory_id, text):
        return {"result": "updated", "memory_id": memory_id}

    def delete(self, memory_id):
        return {"result": "deleted", "memory_id": memory_id}


@pytest.fixture
def qdrant_url():
    pytest.importorskip("qdrant_client")
    from qdrant_client import QdrantClient

    probe = QdrantClient(url=QDRANT_URL)
    try:
        probe.get_collections()  # raises if Qdrant is unreachable
    finally:
        probe.close()
    return QDRANT_URL


def _ensure_collection(client, name):
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config={"size": 8, "distance": "Cosine"},
    )


def test_shutdown_waits_for_inflight_sync(qdrant_url):
    from qdrant_client import QdrantClient

    backend_client = QdrantClient(url=qdrant_url)
    assert_client = QdrantClient(url=qdrant_url)
    try:
        _ensure_collection(assert_client, TEST_COLLECTION)
        backend = SlowQdrantBackend(backend_client, TEST_COLLECTION)

        provider = Mem0MemoryProvider()
        provider._user_id = "shutdown-race-test-user"
        provider._agent_id = "hermes"
        provider._channel = "cli"
        provider._backend = backend

        provider.sync_turn("user likes dark mode", "noted", session_id="s1")
        assert backend.add_entered.wait(10), "sync thread never reached backend.add"

        # sync is mid-`backend.add`; shutdown must wait for it, not close the
        # client out from under it.
        provider.shutdown()

        assert backend.closed_during_add is False, (
            "shutdown() closed the backend while backend.add was still in flight"
        )
        assert backend.last_point_id is not None, (
            "backend.add did not complete its Qdrant insert"
        )
        records = assert_client.retrieve(
            collection_name=TEST_COLLECTION, ids=[backend.last_point_id]
        )
        assert len(records) == 1, (
            "memory point missing from Qdrant after shutdown (lost memory)"
        )
        assert records[0].payload["user_id"] == "shutdown-race-test-user"
    finally:
        try:
            if assert_client.collection_exists(TEST_COLLECTION):
                assert_client.delete_collection(TEST_COLLECTION)
        finally:
            assert_client.close()
            backend_client.close()


def test_shutdown_returns_quickly_when_no_sync_in_flight(qdrant_url):
    """shutdown() must NOT block ~120 s when no sync is in flight."""

    class FastBackend:
        def add(self, messages, *, user_id, agent_id, infer=False, metadata=None):
            return {"results": []}

        def close(self):
            pass

        def search(self, query, *, filters, top_k=10, rerank=False):
            return []

        def update(self, memory_id, text):
            return {"result": "updated"}

        def delete(self, memory_id):
            return {"result": "deleted"}

    provider = Mem0MemoryProvider()
    provider._user_id = "u"
    provider._agent_id = "hermes"
    provider._channel = "cli"
    provider._backend = FastBackend()

    provider.sync_turn("a", "b", session_id="s1")
    # Let the sync thread run to completion so shutdown() observes NO sync in
    # flight and must take the short path.
    provider._sync_thread.join(timeout=5)
    assert not provider._sync_thread.is_alive()

    start = time.monotonic()
    provider.shutdown()
    elapsed = time.monotonic() - start
    # Generous ceiling: catches a regression that always waits the full 120 s,
    # with huge margin over the expected 0.2 s + 5 s joins.
    assert elapsed < 10.0, f"shutdown() blocked {elapsed:.1f}s with no sync in flight"
