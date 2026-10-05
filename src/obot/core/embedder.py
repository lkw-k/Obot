"""bge-m3 dense + sparse embedding (spec §5.5)."""

import functools
import logging
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-m3"
DENSE_DIM = 1024
MAX_LENGTH = 2048
BATCH_SIZE = 16


@dataclass
class Embedding:
    dense: list[float]
    sparse: dict[int, float]


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[Embedding]: ...


class BGEM3Embedder:
    def __init__(self, model: Any = None):
        if model is None:
            # Imported here so tests and non-build commands skip loading torch.
            from FlagEmbedding import BGEM3FlagModel

            model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=False)
        self.model = model

    def encode(self, texts: list[str]) -> list[Embedding]:
        result: list[Embedding] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            self._warn_truncated(batch, start)
            out = self.model.encode(
                batch,
                batch_size=BATCH_SIZE,
                max_length=MAX_LENGTH,
                return_dense=True,
                return_sparse=True,
            )
            for dense, weights in zip(
                out["dense_vecs"], out["lexical_weights"], strict=True
            ):
                result.append(
                    Embedding(
                        dense=[float(x) for x in dense],
                        sparse={int(k): float(v) for k, v in weights.items()},
                    )
                )
            logger.info("embedded %d/%d records", len(result), len(texts))
        return result

    def _warn_truncated(self, batch: list[str], offset: int) -> None:
        lengths = self.model.tokenizer(batch, add_special_tokens=True)["input_ids"]
        for i, ids in enumerate(lengths):
            if len(ids) > MAX_LENGTH:
                logger.warning(
                    "record %d has %d tokens, truncated to %d",
                    offset + i,
                    len(ids),
                    MAX_LENGTH,
                )


@functools.cache
def get_embedder() -> BGEM3Embedder:
    """Load the model once per process."""
    return BGEM3Embedder()
