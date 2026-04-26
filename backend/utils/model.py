import time as _time
from fastembed import TextEmbedding

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

_model: TextEmbedding = None


def get_embedding_model() -> TextEmbedding:
    global _model
    if _model is None:
        _t0 = _time.monotonic()
        _model = TextEmbedding(EMBED_MODEL)
        _elapsed = int((_time.monotonic() - _t0) * 1000)
        print(f"[COLD_START] ONNX model load ({EMBED_MODEL}) took {_elapsed}ms")
    return _model
