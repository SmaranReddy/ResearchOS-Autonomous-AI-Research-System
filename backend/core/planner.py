from typing import List


def get_plan() -> List[str]:
    """
    Returns the ordered list of steps the executor will run.
    Each name must have a corresponding entry in STEP_REGISTRY (executor.py).
    """
    return [
        "search_web",
        "download",
        "preprocess",
        "chunk",
        "embed",
        "index",
        "query_transform",
        "retrieve",
        "rerank",
        "answer",
        "critique",
    ]
