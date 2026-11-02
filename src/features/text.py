"""Text assembly and chunking of selected files."""
from __future__ import annotations


def build_text(files: list[dict], max_chars: int) -> str:
    """Priority-ordered files joined with path headers, hard-capped at max_chars."""
    out, used = [], 0
    for f in files:
        piece = f"### {f['path']}\n{f['text']}\n"
        if used + len(piece) > max_chars:
            out.append(piece[: max_chars - used])
            break
        out.append(piece)
        used += len(piece)
    return "".join(out)


def chunk_text(text: str, chunk_chars: int, overlap: int = 0) -> list[str]:
    """Train-time helper: split `text` into chunks of at most chunk_chars (config features.chunk_chars)."""
    step = max(1, chunk_chars - overlap)
    return [text[i:i + chunk_chars] for i in range(0, max(len(text), 1), step)]
