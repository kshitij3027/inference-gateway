import structlog

logger = structlog.get_logger()


def count_tokens(text: str, model: str) -> int:
    """Count tokens in text using the best available method.

    Strategy: tiktoken (for OpenAI models) -> chars/4 fallback.
    """
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        # Fallback: rough approximation
        return len(text) // 4
