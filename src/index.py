from pathlib import Path

import re
import bm25s
from bm25s.tokenization import Tokenized

from pydantic import TypeAdapter

from tqdm import tqdm

from .data_models import Chunk

# Create corpus
# corpus = [
#     "a cat is a feline and likes to purr",
#     "a dog is the human's best friend and loves to play",
#     "a bird is a beautiful animal that can fly",
#     "a fish is a creature that lives in water and swims",
# ]

# # Tokenize the corpus
# corpus_tokens = bm25s.tokenize(corpus)

# # Index the corpus
# retriever = bm25s.BM25(corpus=corpus)
# retriever.index(corpus_tokens)

# # Query the corpus
# query = "What is an llm?"
# query_tokens = bm25s.tokenize([query])

# # Get top-k results as a tuple of (doc ids, scores)
# results, scores = retriever.retrieve(query_tokens, k=4)

# for i in range(results.shape[1]):
#     doc, score = results[0, i], scores[0, i]
#     print(f"Rank {i+1} (score: {score:.2f}): {doc}")

# # You can save the arrays to a directory...
# retriever.save("vllm_index_bm25")

# # You can save the corpus along with the model
# retriever.save("vllm_index_bm25", corpus=corpus)

# # ...and load them when you need them
# reloaded_retriever = bm25s.BM25.load("vllm_index_bm25", load_corpus=True)
# # set load_corpus=False if you don't need the corpus

# === Read corpus ===

CORPUS_SUFFIXES = {".md", ".txt", ".py"}
CHUNKS_FILENAME = "chunks.json"


def list_corpus_files(raw_dir: str) -> list[Path]:
    """Goes through raw_dir recursively and return a list of files to index"""
    return sorted(f for f in Path(raw_dir).rglob("*")
                  if f.is_file() and f.suffix in CORPUS_SUFFIXES)


def read_file(path: Path) -> str | None:
    """Read a file, return None if not readable (e.g. binary)"""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


# === Chunking ===

DEFAULT_MAX_CHUNK_SIZE: int = 2000


def chunk_fix_size(text: str,
                   file_path: str,
                   max_chunk_size: int,
                   offset: int = 0) -> list[Chunk]:
    """Chunk text following only the max_chunk_size parameter

    offset is the position of text in the original file, so that the
    chunk indices stay absolute.
    """
    chunks: list[Chunk] = []
    pos = 0
    size = len(text)
    while pos < size:
        end = min(pos + max_chunk_size, size)
        cur_chunk = Chunk(file_path=file_path,
                          first_character_index=offset + pos,
                          last_character_index=offset + end,
                          text=text[pos:end])
        chunks.append(cur_chunk)
        pos = end
    return chunks


HEADER_RE = re.compile(r"^#{1,6}(\s|$)")


def is_header(line: str) -> bool:
    """Check if a line is a markdown header (# to ###### then a space)"""
    return HEADER_RE.match(line) is not None


def section_starts(text: str) -> list[int]:
    """Return the starting indices of all sections in the text

    The first section always starts at 0 (text before the first header).
    Ignores headers inside code blocks marked by ``` or ~~~.
    """
    starts = [0]
    pos = 0
    fence = ""
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if fence:
            if stripped.startswith(fence):
                fence = ""
        elif stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
        elif pos > 0 and is_header(line):
            starts.append(pos)
        pos += len(line)
    return starts


def merge_small_sections(starts: list[int], text_length: int,
                         max_chunk_size: int) -> list[tuple[int, int]]:
    """Merge consecutive sections while the result is <= max_chunk_size"""
    merged: list[tuple[int, int]] = []
    ends = starts[1:] + [text_length]
    for start, end in zip(starts, ends):
        if merged:
            prev_start, _ = merged[-1]
            if end - prev_start <= max_chunk_size:
                merged[-1] = (prev_start, end)
                continue
        merged.append((start, end))
    return merged


def section_to_chunks(text: str, file_path: str, start: int, end: int,
                      max_chunk_size: int) -> list[Chunk]:
    """Convert a section of text defined by start and end into chunks"""
    section = text[start:end]
    if not section.strip():
        return []
    if end - start > max_chunk_size:
        return chunk_fix_size(section, file_path, max_chunk_size, offset=start)
    return [
        Chunk(file_path=file_path,
              first_character_index=start,
              last_character_index=end,
              text=section)
    ]


def chunk_sections(
        text: str,
        file_path: str,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE) -> list[Chunk]:
    """Chunk a .md/.txt file according to titles (# ##)"""
    sections = merge_small_sections(section_starts(text), len(text),
                                    max_chunk_size)
    chunks: list[Chunk] = []
    for start, end in sections:
        chunks.extend(
            section_to_chunks(text, file_path, start, end, max_chunk_size))
    return chunks


def chunk_file(path: Path, text: str, max_chunk_size: int) -> list[Chunk]:
    """Pick the chunking strategy according to the file extension"""
    if path.suffix == ".py":
        return chunk_fix_size(text, str(path), max_chunk_size)
    return chunk_sections(text, str(path), max_chunk_size)


# === Tokenization ===


def tokenize(texts: list[str],
             show_progress: bool = False) -> list[list[str]] | Tokenized:
    """Transform texts into tokens for BM25, also used in retrieval.py"""
    return bm25s.tokenize(texts, stopwords="en", show_progress=show_progress)


# === Construct ===


def build_index(raw_dir: str,
                output_dir: str,
                max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE) -> None:
    """files -> chunks (tqdm) -> tokens -> BM25 -> save"""
    chunks: list[Chunk] = []
    for file in tqdm(list_corpus_files(raw_dir), desc="Chunking"):
        text = read_file(file)
        if text is None:
            continue
        chunks.extend(chunk_file(file, text, max_chunk_size))
    if not chunks:
        raise ValueError(f"No chunk to index in {raw_dir!r}")
    retriever = bm25s.BM25()
    retriever.index(tokenize([c.text for c in chunks], show_progress=True))
    save_index(retriever, chunks, output_dir)


CHUNKS_ADAPTER = TypeAdapter(list[Chunk])


def save_index(retriever: bm25s.BM25, chunks: list[Chunk],
               output_dir: str) -> None:
    """Save BM25 index and chunks metadatas (json) in output_dir"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    retriever.save(str(out))
    (out / CHUNKS_FILENAME).write_bytes(CHUNKS_ADAPTER.dump_json(chunks))


def load_index(output_dir: str) -> tuple[bm25s.BM25, list[Chunk]]:
    """Reload index + chunks. Used by retrieval.py

    The i-th document of the BM25 index is chunks[i].
    """
    retriever = bm25s.BM25.load(output_dir)
    raw = (Path(output_dir) / CHUNKS_FILENAME).read_bytes()
    return retriever, CHUNKS_ADAPTER.validate_json(raw)
