"""Standalone inspection script: chunk BGE and print stats + samples."""

from pathlib import Path
from ingest.chunk import chunk_work, _token_count

RAW = Path("data/raw/beyond_good_and_evil.txt")

chunks = chunk_work(
    RAW.read_text(encoding="utf-8"),
    "Beyond Good and Evil",
    "beyond_good_and_evil",
    "late",
    "aphorism",
)

token_counts = [_token_count(c.content) for c in chunks]
avg = sum(token_counts) / len(token_counts)

print(f"\nTotal chunks : {len(chunks)}")
print(f"Avg tokens   : {avg:.1f}")
print(f"Min / Max    : {min(token_counts)} / {max(token_counts)}")

# 10 evenly-spaced samples
step = max(1, len(chunks) // 10)
samples = chunks[::step][:10]

print(f"\n{'─' * 72}")
for c in samples:
    aph = (
        f"§{c.aphorism_number}"
        if c.aphorism_number == c.aphorism_number_end
        else f"§{c.aphorism_number}–{c.aphorism_number_end}"
    )
    label = f"[ch.{c.section_number} {aph}]" if c.aphorism_number else "[preamble]"
    tokens = _token_count(c.content)
    preview = " ".join(c.content.split()[:40])
    print(f"\nchunk {c.chunk_index:>3}  {label}  ({tokens} tokens)")
    print(f"  {preview}{'…' if tokens > 40 else ''}")

print(f"\n{'─' * 72}")
