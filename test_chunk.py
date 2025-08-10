import os
import re
import uuid
import json

from pathlib import Path

# Settings
root_dir = Path("data/raw/_sources")  # You can upload the docs ZIP and unzip here
output_path = Path("data/chunked/chunks.json")

# Chunking thresholds
MAX_CHUNK_TOKENS = 500
MIN_CHUNK_TOKENS = 100  # Smaller chunks will be merged if under this

# Header pattern for reStructuredText sections (e.g., 'Title\n=====' or 'Section\n-----')
HEADER_RE = re.compile(r"^(?P<title>[^\n]+)\n(?P<underline>=+|-+|~+)$", re.MULTILINE)

# Helper to estimate token count (simple approximation)
def estimate_tokens(text):
    return len(text.split())

# Clean and normalize text
def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

# Chunk a single file
def chunk_rst_file(filepath, relpath):
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    matches = list(HEADER_RE.finditer(content))
    chunks = []

    if not matches:
        # Fallback: single chunk
        text = clean_text(content)
        if len(text) > 0:
            chunks.append({
                "id": str(uuid.uuid4()),
                "title": relpath.stem,
                "text": text,
                "file_path": str(relpath),
                "section": relpath.parts[0],
                "subsection": relpath.parts[1] if len(relpath.parts) > 2 else None,
            })
        return chunks

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        chunk_text = clean_text(content[start:end])
        if chunk_text:
            chunks.append({
                "id": str(uuid.uuid4()),
                "title": match.group("title").strip(),
                "text": chunk_text,
                "file_path": str(relpath),
                "section": relpath.parts[0],
                "subsection": relpath.parts[1] if len(relpath.parts) > 2 else None,
            })
    return chunks

# Walk the directory and collect all .rst.txt files
all_chunks = []

for dirpath, _, filenames in os.walk(root_dir):
    for fname in filenames:
        if fname.endswith(".rst.txt"):
            fpath = Path(dirpath) / fname
            relpath = fpath.relative_to(root_dir)
            chunks = chunk_rst_file(fpath, relpath)
            all_chunks.extend(chunks)

# Save to JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)
