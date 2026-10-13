import json
from pathlib import Path

import re
import bm25s
from bm25s.tokenization import Tokenized

from tqdm import tqdm

from .data_models import MinimalSource

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
                   offset: int = 0) -> list[MinimalSource]:
    """Chunk text following only the max_chunk_size parameter

    offset is the position of text in the original file, so that the
    chunk indices stay absolute.
    """
    chunks: list[MinimalSource] = []
    pos = 0
    size = len(text)
    while pos < size:
        end = min(pos + max_chunk_size, size)
        cur_chunk = MinimalSource(file_path=file_path,
                                  first_character_index=offset + pos,
                                  last_character_index=offset + end,
                                  text=text[pos:end])
        chunks.append(cur_chunk)
        pos = end
    return chunks


def is_header(line: str) -> bool:
    """Check if a line is a markdown header"""
    return line.lstrip().startswith("#")


def section_starts(text: str) -> list[int]:
    """Return the starting indices of all sections in the text

    The first section always starts at 0 (text before the first header).
    Ignores headers inside code blocks marked by ```.
    """
    starts = [0]
    pos = 0
    fence = ""
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if fence:
            if stripped.startswith(fence):
                fence = ""
        elif stripped.startswith("```"):
            fence = "```"
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
                      max_chunk_size: int) -> list[MinimalSource]:
    """Convert a section of text defined by start and end into chunks"""
    section = text[start:end]
    if not section.strip():
        return []
    if end - start > max_chunk_size:
        return chunk_fix_size(section, file_path, max_chunk_size, offset=start)
    return [
        MinimalSource(file_path=file_path,
                      first_character_index=start,
                      last_character_index=end,
                      text=section)
    ]


def chunk_sections(
        text: str,
        file_path: str,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE) -> list[MinimalSource]:
    """Chunk a .md/.txt file according to titles (# ##)"""
    sections = merge_small_sections(section_starts(text), len(text),
                                    max_chunk_size)
    chunks: list[MinimalSource] = []
    for start, end in sections:
        chunks.extend(
            section_to_chunks(text, file_path, start, end, max_chunk_size))
    return chunks


def chunk_file(path: Path, text: str,
               max_chunk_size: int) -> list[MinimalSource]:
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
    """Main function to build the BM25 index from raw text files.

    Stages:
    1. Chunk raw text files into sections.
    2. Tokenize the chunks.
    3. Build the BM25 index.
    4. Save the index and chunk metadata.
    """
    chunks: list[MinimalSource] = []
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


# === Save and Load Index ===


def save_index(retriever: bm25s.BM25, chunks: list[MinimalSource],
               output_dir: str) -> None:
    """Save BM25 index and chunks metadatas (json) in output_dir"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    retriever.save(str(out))
    data = [chunk.model_dump() for chunk in chunks]
    (out / CHUNKS_FILENAME).write_text(json.dumps(data), encoding="utf-8")


def load_index(output_dir: str) -> tuple[bm25s.BM25, list[MinimalSource]]:
    """Reload index + chunks. Used by retrieval.py

    The i-th document of the BM25 index is chunks[i].
    """
    retriever = bm25s.BM25.load(output_dir)
    raw = (Path(output_dir) / CHUNKS_FILENAME).read_text(encoding="utf-8")
    chunks = [MinimalSource(**data) for data in json.loads(raw)]
    return retriever, chunks
