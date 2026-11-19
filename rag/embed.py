"""Embeddings through Cloudflare Workers AI, for the deployed app.

The index was built locally with Ollama's qwen3-embedding:0.6b. Cloudflare
hosts the same model, and its vectors match Ollama's to four decimal places,
so the deployed app can search the locally built index without re-indexing.
EMBEDDINGS=cloudflare (the default) uses this; EMBEDDINGS=ollama uses the
local model instead.
"""
import os

import requests
from langchain_core.embeddings import Embeddings

from . import config

MODEL = "@cf/qwen/qwen3-embedding-0.6b"
BATCH = 100


class CloudflareEmbeddings(Embeddings):
    def __init__(self):
        if not config.CF_TOKEN:
            raise RuntimeError("EMBEDDINGS=cloudflare needs CLOUDFLARE_API_TOKEN. See example.env.")
        self.headers = {"Authorization": f"Bearer {config.CF_TOKEN}"}
        account = config.CF_ACCOUNT_ID
        if not account:
            r = requests.get("https://api.cloudflare.com/client/v4/accounts",
                             headers=self.headers, timeout=30).json()
            account = r["result"][0]["id"]
        self.url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{MODEL}"

    def embed_documents(self, texts):
        out = []
        for start in range(0, len(texts), BATCH):
            r = requests.post(self.url, headers=self.headers,
                              json={"text": texts[start:start + BATCH]}, timeout=120).json()
            if not r.get("success"):
                raise RuntimeError(f"Cloudflare embeddings refused: {r.get('errors')}")
            out.extend(r["result"]["data"])
        return out

    def embed_query(self, text):
        return self.embed_documents([text])[0]
