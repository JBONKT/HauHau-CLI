"""Optional embedding client.

The generation model cannot embed: llama-server reports
``Pooling type 'none' is not OAI compatible``. A second tiny server
(``llama-server -m <embed.gguf> --embeddings``) is the supported path. When no
embedding endpoint is configured the brain runs keyword-only, which is fully
functional.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class Embedder:
    def __init__(self, url: str, model: str = "", timeout: float = 30.0) -> None:
        self.url = url.strip()
        self.model = model
        self.timeout = timeout
        self._broken = False
        self.last_error = ""

    @property
    def enabled(self) -> bool:
        return bool(self.url) and not self._broken

    def embed(self, text: str) -> list[float] | None:
        if not self.enabled or not text.strip():
            return None
        payload = json.dumps({"input": text[:4000], "model": self.model or "local"}).encode()
        req = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            vector = data["data"][0]["embedding"]
            return [float(x) for x in vector]
        except (urllib.error.URLError, KeyError, IndexError, ValueError, TimeoutError) as exc:
            close = getattr(exc, "close", None)
            if callable(close):
                close()
            # One failure disables vectors for the session instead of retrying
            # on every turn; keyword retrieval keeps working.
            self._broken = True
            self.last_error = str(exc)
            return None
