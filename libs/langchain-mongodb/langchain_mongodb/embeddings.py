from __future__ import annotations

import re

from langchain_core.embeddings import Embeddings

# either return not implemented error or import voyage ai's
class AutoEmbedding(Embeddings):
    def __init__(self, model_name: str):
        # GOAL: check for supported models
        # are all voyage models going to start with "voyage-x.x"?
        # should I regex search for model name? or what's the best way to do this going forward?
        self.model_name = model_name
        match = re.search(r'voyage-(\d+)', model_name)
        major = int(match.group(1))
        # TODO: fix this logic!
        if major < 3:
            print("supported voyage models are voyage-4")
            return

    # TODO: we should implement the API and base it on voyage's but don't copy
    #  paste cuz we may allow for non-voyage moduls in the future

    # okay i understand why this is here, but also it feels silly for an
    # auto-embedding to actually have methods that embed docs when it'll never be called?
    # would it make more sense to just return an empty list since this class isn't actually doing the embeddings?
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # """Embed search docs."""
        return texts

    def embed_query(self, text: str) -> List[float]:
        # """Embed query text."""
        return text

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        # """Async embed search docs."""
        return texts

    async def aembed_query(self, text: str) -> List[float]:
        # """Async embed query text."""
        return text