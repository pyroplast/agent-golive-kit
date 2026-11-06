"""Knowledge base: load markdown pages, split by heading, keyword search."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

STOPWORDS = set("""a an and are as at be by can do does for from has have how i in is it its
of on or our the this to we what when where which who will with you your""".split())


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    title: str
    url: str
    heading: str
    text: str


def _front_matter(raw: str) -> tuple[dict, str]:
    meta: dict = {}
    if raw.startswith("---"):
        _, head, body = raw.split("---", 2)
        for line in head.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        return meta, body
    return meta, raw


def load_kb(path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for file in sorted(Path(path).glob("*.md")):
        meta, body = _front_matter(file.read_text(encoding="utf-8"))
        sections = re.split(r"^## +", body, flags=re.M)
        n = 0
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            heading, _, text = sec.partition("\n")
            n += 1
            chunks.append(Chunk(
                id=f"{file.stem}#{n}", doc=file.stem,
                title=meta.get("title", file.stem), url=meta.get("url", ""),
                heading=heading.strip(), text=text.strip(),
            ))
    return chunks


def format_kb(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[{c.id}] {c.title} > {c.heading} ({c.url})\n{c.text}" for c in chunks
    )


def tokenize(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOPWORDS:
            continue
        for suffix in ("ing", "ed", "es", "s"):
            if len(w) > 4 and w.endswith(suffix):
                w = w[: -len(suffix)]
                break
        out.append(w)
    return out


class BM25Index:
    """Plain BM25. Used when the knowledge base is too big to fit in the prompt."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(f"{c.heading} {c.text}")) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        scored = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in terms:
                if t not in d:
                    continue
                tf = d[t]
                s += self.idf[t] * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            if s > 0:
                scored.append((self.chunks[i], s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
