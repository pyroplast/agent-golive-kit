from deploykit.kb import BM25Index, format_kb, load_kb


def test_load_kb_chunks_by_heading(chunks):
    ids = [c.id for c in chunks]
    assert ids == ["pricing#1", "pricing#2", "security#1", "security#2"]
    first = chunks[0]
    assert first.title == "Pricing"
    assert first.url == "https://example.com/pricing"
    assert first.heading == "Pricing model"
    assert "quote-based" in first.text
    assert "---" not in first.text


def test_load_kb_ignores_non_markdown(tmp_path):
    (tmp_path / "a.md").write_text("---\ntitle: A\nurl: u\n---\n## H\nbody\n")
    (tmp_path / "notes.txt").write_text("ignore me")
    assert [c.id for c in load_kb(tmp_path)] == ["a#1"]


def test_format_kb_labels_every_chunk(chunks):
    text = format_kb(chunks)
    for c in chunks:
        assert f"[{c.id}]" in text
    assert "Ireland" in text


def test_bm25_ranks_relevant_chunk_first(chunks):
    index = BM25Index(chunks)
    results = index.search("which AWS region hosts the data", k=2)
    assert results[0][0].id == "security#2"
    assert 1 <= len(results) <= 2
    assert all(score > 0 for _, score in results)
    assert [s for _, s in results] == sorted((s for _, s in results), reverse=True)


def test_bm25_no_match_returns_empty(chunks):
    assert BM25Index(chunks).search("zebra quantum", k=3) == []
