from __future__ import annotations

from skillPilot.skills import SkillStore, _cosine_similarity, _skill_text


def _mock_embed_fn(texts: list[str]) -> list[list[float]]:
    """Deterministic mock embedding: map each character to a dimension, normalise."""
    results = []
    for text in texts:
        vec = [0.0] * 128
        for ch in text.lower():
            if ord(ch) < 128:
                vec[ord(ch)] += 1.0
        # Normalise
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        results.append(vec)
    return results


def test_cosine_similarity_identical_vectors():
    assert abs(_cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal():
    assert abs(_cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_cosine_similarity_opposite():
    assert abs(_cosine_similarity([1.0, 0.0], [-1.0, 0.0]) - (-1.0)) < 1e-9


def test_cosine_similarity_empty_vectors():
    assert _cosine_similarity([], []) == 0.0


def test_cosine_similarity_different_lengths():
    assert _cosine_similarity([1.0, 2.0], [1.0]) == 0.0


def test_cosine_similarity_zero_norm():
    assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_skill_text_composition():
    from skillPilot.models import SkillRecord
    skill = SkillRecord(
        id="test_001",
        name="fast_api_builder",
        description="Create FastAPI applications",
        body="Use uvicorn and FastAPI to build REST APIs",
        tags=["python", "fastapi", "web"],
        source_task="build a web service",
    )
    text = _skill_text(skill)
    assert "fast_api_builder" in text
    assert "Create FastAPI applications" in text
    assert "python" in text
    assert "build a web service" in text


def test_search_without_embed_fn_falls_back_to_lexical(tmp_path):
    """Without embed_fn, search should still work via lexical matching."""
    store = SkillStore(tmp_path / "skills")
    store.create(
        name="python_cli_script",
        description="Create Python CLI scripts",
        body="Write a .py file and run it",
        tags=["python", "cli"],
    )
    results = store.search("python script cli", top_k=3)
    assert len(results) > 0
    assert results[0][0].name == "python_cli_script"
    # No embedding stored since no embed_fn
    assert results[0][0].embedding == []


def test_search_with_embed_fn_uses_hybrid_scoring(tmp_path):
    """With embed_fn, search should use semantic similarity."""
    store = SkillStore(tmp_path / "skills", embed_fn=_mock_embed_fn)
    store.create(
        name="http_server_setup",
        description="Set up HTTP web servers and REST APIs",
        body="Use Flask or FastAPI to create HTTP endpoints",
        tags=["web", "api", "server"],
        source_task="build a web service",
    )
    store.create(
        name="file_sorting_utility",
        description="Sort and organize files by extension",
        body="Use pathlib and shutil to move files into folders",
        tags=["filesystem", "utility"],
        source_task="organize my downloads folder",
    )
    # Query with synonym "web framework" — should find "http_server_setup" via semantics
    results = store.search("web framework server API", top_k=5)
    assert len(results) >= 1
    assert results[0][0].name == "http_server_setup"


def test_create_generates_embedding(tmp_path):
    """create() should generate and store an embedding when embed_fn is available."""
    store = SkillStore(tmp_path / "skills", embed_fn=_mock_embed_fn)
    skill = store.create(
        name="test_skill",
        description="A test skill",
        body="Test body content",
        tags=["test"],
    )
    assert len(skill.embedding) == 128  # Mock produces 128-dim vectors
    assert any(v != 0.0 for v in skill.embedding)


def test_create_without_embed_fn_no_embedding(tmp_path):
    """create() without embed_fn should leave embedding empty."""
    store = SkillStore(tmp_path / "skills")
    skill = store.create(
        name="test_skill",
        description="A test skill",
        body="Test body content",
        tags=["test"],
    )
    assert skill.embedding == []


def test_update_regenerates_embedding(tmp_path):
    """update() should regenerate embedding when content changes."""
    store = SkillStore(tmp_path / "skills", embed_fn=_mock_embed_fn)
    skill = store.create(
        name="original_skill",
        description="Original description",
        body="Original body",
        tags=["original"],
    )
    original_embedding = list(skill.embedding)

    updated = store.update(
        skill.id,
        body="Completely different body content now",
        reason="Changed purpose",
    )
    assert updated.embedding != original_embedding
    assert len(updated.embedding) == 128


def test_skill_without_embedding_falls_back_to_lexical(tmp_path):
    """Skills created without embed_fn should still be found via lexical when queried with embed_fn."""
    store_no_embed = SkillStore(tmp_path / "skills")
    store_no_embed.create(
        name="python_script_writer",
        description="Write Python scripts",
        body="Create .py files and run them",
        tags=["python", "script"],
    )
    # Now search with a store that HAS embed_fn — old skill has no embedding
    store_with_embed = SkillStore(tmp_path / "skills", embed_fn=_mock_embed_fn)
    results = store_with_embed.search("python script", top_k=3)
    assert len(results) > 0
    assert results[0][0].name == "python_script_writer"


def test_embed_fn_failure_graceful_degradation(tmp_path):
    """If embed_fn raises, create/search should still work via lexical fallback."""
    call_count = {"create": 0, "search": 0}

    def failing_embed(texts):
        call_count["create" if len(texts) == 1 else "search"] += 1
        raise RuntimeError("Embedding API unavailable")

    store = SkillStore(tmp_path / "skills", embed_fn=failing_embed)
    skill = store.create(
        name="fallback_skill",
        description="A skill for testing fallback",
        body="Body content for fallback test",
        tags=["fallback"],
    )
    assert skill.embedding == []  # No embedding stored due to failure

    results = store.search("fallback test", top_k=3)
    assert len(results) > 0
    assert results[0][0].name == "fallback_skill"


def test_success_rate_bonus_affects_ranking(tmp_path):
    """Skills with higher success rates should rank slightly higher."""
    store = SkillStore(tmp_path / "skills", embed_fn=_mock_embed_fn)

    # Create two similar skills
    skill_a = store.create(
        name="api_endpoint_a",
        description="Build REST API endpoints",
        body="Create HTTP endpoints with Flask",
        tags=["api", "web"],
    )
    skill_b = store.create(
        name="api_endpoint_b",
        description="Build REST API endpoints",
        body="Create HTTP endpoints with FastAPI",
        tags=["api", "web"],
    )

    # Simulate usage: skill_a succeeds 10/10, skill_b succeeds 7/10
    # Use alternating pattern to avoid consecutive failures triggering deprecation
    for _ in range(10):
        store.record_usage(skill_a.id, success=True)
    # Pattern: S F S F S F S F S S = 7 success, 3 fail (no 3 consecutive fails)
    pattern_b = [True, False, True, False, True, False, True, False, True, True]
    for success in pattern_b:
        store.record_usage(skill_b.id, success=success)

    # Re-read store to pick up updated usage counts
    store2 = SkillStore(tmp_path / "skills", embed_fn=_mock_embed_fn)
    results = store2.search("REST API endpoints", top_k=5)
    assert len(results) >= 2, f"Expected at least 2 results, got {len(results)}"
    # skill_a should rank higher due to success rate bonus
    ids = [r[0].id for r in results]
    assert ids.index(skill_a.id) < ids.index(skill_b.id), (
        f"skill_a should rank higher than skill_b, but got order: {ids}"
    )


def test_skill_create_search_update_usage(tmp_path):
    store = SkillStore(tmp_path / "skills")
    skill = store.create(
        name="create_python_cli_script",
        description="Create and validate small Python CLI scripts",
        body="Write a .py file with an if __name__ == '__main__' entrypoint, then run it.",
        tags=["python", "cli"],
        source_task="make a script",
    )

    results = store.search("write python command line script", top_k=3)
    assert results
    assert results[0][0].id == skill.id

    updated = store.update(
        skill.id,
        body="Write a .py file with an entrypoint, run it, and add pytest coverage if logic is reusable.",
        tags=["pytest"],
        reason="Captured validation pattern",
    )
    assert updated.version == "1.0.1"
    assert "pytest" in updated.tags

    store.record_usage(skill.id, success=True)
    used = store.get(skill.id)
    assert used is not None
    assert used.usage_count == 1
    assert used.success_count == 1
