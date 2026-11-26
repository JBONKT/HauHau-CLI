from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

from hauhau.brain import Brain
from hauhau.brain.embed import Embedder
from hauhau.config import Config


class _Handler(BaseHTTPRequestHandler):
    """Minimal stand-in for `llama-server --embeddings`."""

    vector: ClassVar[list[float]] = [0.1, 0.2, 0.3]
    fail: ClassVar[bool] = False
    last_body: ClassVar[dict] = {}

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        type(self).last_body = json.loads(self.rfile.read(length) or b"{}")
        if type(self).fail:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        body = json.dumps({"data": [{"embedding": self.vector}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


class EmbedderTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._server = HTTPServer(("127.0.0.1", 0), _Handler)
        cls.url = f"http://127.0.0.1:{cls._server.server_port}/v1/embeddings"
        cls._thread = threading.Thread(target=cls._server.serve_forever, daemon=True)
        cls._thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server.shutdown()
        cls._server.server_close()

    def setUp(self) -> None:
        _Handler.fail = False
        _Handler.vector = [0.1, 0.2, 0.3]
        _Handler.last_body = {}


class TestEmbedder(EmbedderTestCase):
    def test_returns_a_vector(self) -> None:
        self.assertEqual(Embedder(self.url, "m").embed("hello"), [0.1, 0.2, 0.3])

    def test_request_carries_input_and_model(self) -> None:
        Embedder(self.url, "embeddinggemma-300m").embed("hello")
        self.assertEqual(_Handler.last_body["input"], "hello")
        self.assertEqual(_Handler.last_body["model"], "embeddinggemma-300m")

    def test_empty_text_skips_the_request(self) -> None:
        self.assertIsNone(Embedder(self.url, "m").embed("   "))
        self.assertEqual(_Handler.last_body, {})

    def test_no_url_means_disabled(self) -> None:
        embedder = Embedder("")
        self.assertFalse(embedder.enabled)
        self.assertIsNone(embedder.embed("hello"))

    def test_server_error_disables_instead_of_retrying(self) -> None:
        _Handler.fail = True
        embedder = Embedder(self.url, "m", timeout=5)
        self.assertIsNone(embedder.embed("hello"))
        self.assertFalse(embedder.enabled)
        self.assertTrue(embedder.last_error)
        # A second call must not even try, and must not raise.
        self.assertIsNone(embedder.embed("again"))

    def test_text_is_capped_before_sending(self) -> None:
        Embedder(self.url, "m").embed("x" * 10_000)
        self.assertLessEqual(len(_Handler.last_body["input"]), 4000)


class TestBrainWithVectors(EmbedderTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.cfg = Config()
        self.cfg.logging.data_dir = str(Path(self._tmp.name) / "data")
        self.cfg.brain.embeddings_url = self.url
        self.brain = Brain(self.cfg)

    def tearDown(self) -> None:
        self.brain.close()
        self._tmp.cleanup()

    def test_remember_stores_a_vector(self) -> None:
        self.brain.remember("semantic", "The deploy script lives in bin/release.")
        self.assertEqual(self.brain.stats()["with_embeddings"], 1)
        self.assertTrue(self.brain.stats()["embeddings_enabled"])

    def test_recall_embeds_the_query(self) -> None:
        self.brain.remember("semantic", "The deploy script lives in bin/release.")
        _Handler.last_body = {}
        self.brain.recall("where is the deploy script")
        self.assertEqual(_Handler.last_body["input"], "where is the deploy script")

    def test_recall_works_after_the_endpoint_dies(self) -> None:
        self.brain.remember("procedural", "Run make release to deploy.")
        _Handler.fail = True
        hits = self.brain.recall("deploy command")
        self.assertTrue(hits, "keyword retrieval must survive a vector outage")


if __name__ == "__main__":
    unittest.main()
