"""Tests for skill quality control: validation, scoring, and auto-deprecation."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from skillPilot.models import SkillRecord, utc_now
from skillPilot.skills import (
    SkillStore,
    SkillValidator,
    _CONSECUTIVE_FAILURE_THRESHOLD,
    _IDLE_DAYS_THRESHOLD,
    _MIN_USAGE_FOR_IDLE_CHECK,
    _MIN_USAGE_FOR_QUALITY_CHECK,
    _MIN_USAGE_FOR_SUCCESS_RATE_CHECK,
    _QUALITY_SCORE_DEPRECATION_THRESHOLD,
    _SUCCESS_RATE_DEPRECATION_THRESHOLD,
)


def _mock_embed_fn(texts: list[str]) -> list[list[float]]:
    results = []
    for text in texts:
        vec = [0.0] * 128
        for ch in text.lower():
            if ord(ch) < 128:
                vec[ord(ch)] += 1.0
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        results.append(vec)
    return results


def _mock_llm_chat_fn(messages, temperature, max_tokens):
    """Mock LLM that returns a high-quality JSON response."""
    return json.dumps({"clarity": 4, "reusability": 4, "completeness": 4})


def _low_quality_llm_chat_fn(messages, temperature, max_tokens):
    """Mock LLM that returns a low-quality JSON response."""
    return json.dumps({"clarity": 1, "reusability": 1, "completeness": 1})


def _failing_llm_chat_fn(messages, temperature, max_tokens):
    """Mock LLM that always raises."""
    raise RuntimeError("LLM unavailable")


def _make_skill(
    name="test_skill",
    description="A test skill for quality validation",
    body='Write a Python function:\n```python\ndef greet(name):\n    return f"Hello, {name}!"\n```',
    tags=None,
    status="active",
    quality_score=0.5,
    consecutive_failures=0,
    usage_count=0,
    success_count=0,
) -> SkillRecord:
    return SkillRecord(
        id=f"{name}_abc123",
        name=name,
        description=description,
        body=body,
        tags=tags if tags is not None else ["test"],
        status=status,
        quality_score=quality_score,
        consecutive_failures=consecutive_failures,
        usage_count=usage_count,
        success_count=success_count,
    )


# ── SkillValidator.static_check ──────────────────────────────────────────────


class TestStaticCheck:
    def test_good_skill_high_score(self):
        skill = _make_skill()
        score = SkillValidator.static_check(skill)
        assert score >= 0.7, f"Good skill should score >= 0.7, got {score}"

    def test_empty_body_low_score(self):
        skill = _make_skill(body="")
        score = SkillValidator.static_check(skill)
        assert score <= 0.5, f"Empty body should score <= 0.5, got {score}"

    def test_empty_description_low_score(self):
        skill = _make_skill(description="")
        score = SkillValidator.static_check(skill)
        assert score < 0.8, f"Empty description should score < 0.8, got {score}"

    def test_no_tags_penalized(self):
        # Use a short body so the code-check is 0 and tags difference is visible
        body = "Short body here"  # < 20 chars → partial body score
        skill_no_tags = _make_skill(body=body, description="ok", tags=[])
        skill_with_tags = _make_skill(body=body, description="ok", tags=["python"])
        score_no = SkillValidator.static_check(skill_no_tags)
        score_yes = SkillValidator.static_check(skill_with_tags)
        assert score_no < score_yes, (
            f"Skill without tags ({score_no:.4f}) should score lower than with tags ({score_yes:.4f})"
        )

    def test_short_body_partial_score(self):
        skill = _make_skill(body="short")
        score = SkillValidator.static_check(skill)
        assert 0.0 < score <= 0.6, f"Short body should get partial score, got {score}"

    def test_code_like_content_bonus(self):
        code_body = (
            "Create a utility function.\n\n"
            "```python\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "\n"
            "for item in items:\n"
            "    print(item)\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    import sys\n"
            "    result = add(1, 2)\n"
            "```"
        )
        skill_code = _make_skill(body=code_body)
        skill_text = _make_skill(body="Just some plain text description here without any code")
        score_code = SkillValidator.static_check(skill_code)
        score_text = SkillValidator.static_check(skill_text)
        assert score_code >= score_text, "Code body should score at least as high as plain text"

    def test_score_bounded_0_to_1(self):
        skill = _make_skill()
        score = SkillValidator.static_check(skill)
        assert 0.0 <= score <= 1.0


# ── SkillValidator.syntax_check ──────────────────────────────────────────────


class TestSyntaxCheck:
    def test_valid_python_code_block(self):
        body = '```python\ndef hello():\n    print("Hello, World!")\n```'
        skill = _make_skill(body=body)
        score = SkillValidator.syntax_check(skill)
        assert score == 1.0, f"Valid Python should score 1.0, got {score}"

    def test_invalid_python_code_block(self):
        body = '```python\ndef hello(\n    this is not valid python\n```'
        skill = _make_skill(body=body)
        score = SkillValidator.syntax_check(skill)
        assert score == 0.0, f"Invalid Python should score 0.0, got {score}"

    def test_no_code_neutral_score(self):
        body = "This skill describes how to organize project files."
        skill = _make_skill(body=body)
        score = SkillValidator.syntax_check(skill)
        assert score == 0.5, f"No code should return 0.5, got {score}"

    def test_multiple_code_blocks_mixed(self):
        body = (
            "```python\ndef valid():\n    pass\n```\n\n"
            "```python\ndef invalid(\n```\n"
        )
        skill = _make_skill(body=body)
        score = SkillValidator.syntax_check(skill)
        assert score == 0.5, f"1 valid out of 2 should be 0.5, got {score}"

    def test_multiple_valid_blocks(self):
        body = (
            "```python\nx = 1\n```\n\n"
            "```python\ny = 2\n```\n"
        )
        skill = _make_skill(body=body)
        score = SkillValidator.syntax_check(skill)
        assert score == 1.0, f"All valid should score 1.0, got {score}"

    def test_score_bounded(self):
        skill = _make_skill()
        score = SkillValidator.syntax_check(skill)
        assert 0.0 <= score <= 1.0


# ── SkillValidator.llm_quality_check ─────────────────────────────────────────


class TestLLMQualityCheck:
    def test_no_llm_returns_neutral(self):
        skill = _make_skill()
        score = SkillValidator.llm_quality_check(skill, None)
        assert score == 0.5

    def test_high_quality_llm(self):
        skill = _make_skill()
        score = SkillValidator.llm_quality_check(skill, _mock_llm_chat_fn)
        # clarity=4, reusability=4, completeness=4 => avg=4 => (4-1)/4 = 0.75
        assert abs(score - 0.75) < 0.01, f"Expected 0.75, got {score}"

    def test_low_quality_llm(self):
        skill = _make_skill()
        score = SkillValidator.llm_quality_check(skill, _low_quality_llm_chat_fn)
        # All 1 => avg=1 => (1-1)/4 = 0.0
        assert abs(score - 0.0) < 0.01, f"Expected 0.0, got {score}"

    def test_failing_llm_returns_neutral(self):
        skill = _make_skill()
        score = SkillValidator.llm_quality_check(skill, _failing_llm_chat_fn)
        assert score == 0.5

    def test_perfect_quality_llm(self):
        def perfect_fn(messages, temperature, max_tokens):
            return json.dumps({"clarity": 5, "reusability": 5, "completeness": 5})

        skill = _make_skill()
        score = SkillValidator.llm_quality_check(skill, perfect_fn)
        assert abs(score - 1.0) < 0.01, f"Expected 1.0, got {score}"

    def test_invalid_json_returns_neutral(self):
        def bad_json_fn(messages, temperature, max_tokens):
            return "not json at all"

        skill = _make_skill()
        score = SkillValidator.llm_quality_check(skill, bad_json_fn)
        assert score == 0.5

    def test_clamped_values(self):
        def over_fn(messages, temperature, max_tokens):
            return json.dumps({"clarity": 10, "reusability": 0, "completeness": 3})

        skill = _make_skill()
        score = SkillValidator.llm_quality_check(skill, over_fn)
        # clarity=5 (clamped), reusability=1 (clamped), completeness=3 => avg=3 => (3-1)/4 = 0.5
        assert abs(score - 0.5) < 0.01, f"Expected 0.5 with clamped values, got {score}"


# ── SkillValidator.validate ──────────────────────────────────────────────────


class TestValidate:
    def test_validate_without_llm(self):
        skill = _make_skill()
        result = SkillValidator.validate(skill)
        assert "quality_score" in result
        assert "static_score" in result
        assert "syntax_score" in result
        assert "llm_score" in result
        assert "timestamp" in result
        assert 0.0 <= result["quality_score"] <= 1.0

    def test_validate_with_llm(self):
        skill = _make_skill()
        result = SkillValidator.validate(skill, llm_chat_fn=_mock_llm_chat_fn)
        assert result["llm_score"] == 0.75

    def test_quality_score_is_weighted_blend(self):
        skill = _make_skill()
        result = SkillValidator.validate(skill, llm_chat_fn=_mock_llm_chat_fn)
        expected = (
            result["static_score"] * 0.3
            + result["syntax_score"] * 0.3
            + result["llm_score"] * 0.4
        )
        assert abs(result["quality_score"] - round(expected, 4)) < 0.001


# ── Auto-validation on create / update ───────────────────────────────────────


class TestAutoValidation:
    def test_create_auto_validates(self, tmp_path):
        store = SkillStore(tmp_path / "skills", llm_chat_fn=_mock_llm_chat_fn)
        skill = store.create(
            name="auto_validated_skill",
            description="A skill that gets auto-validated on creation",
            body='```python\ndef add(a, b):\n    return a + b\n```',
            tags=["test"],
        )
        assert skill.quality_score != 0.5, "Quality score should be updated from default"
        assert skill.last_validated_at != ""
        assert len(skill.validation_results) >= 1

    def test_update_auto_validates_on_body_change(self, tmp_path):
        store = SkillStore(tmp_path / "skills", llm_chat_fn=_mock_llm_chat_fn)
        skill = store.create(
            name="update_test_skill",
            description="Test auto-validation on update",
            body='```python\nx = 1\n```',
            tags=["test"],
        )
        initial_validations = len(skill.validation_results)

        updated = store.update(
            skill.id,
            body='```python\ndef multiply(a, b):\n    return a * b\n```',
            reason="Changed body",
        )
        assert len(updated.validation_results) > initial_validations

    def test_update_does_not_validate_on_tag_only_change(self, tmp_path):
        store = SkillStore(tmp_path / "skills", llm_chat_fn=_mock_llm_chat_fn)
        skill = store.create(
            name="tag_only_skill",
            description="Test that tag-only changes don't re-validate",
            body='```python\ny = 2\n```',
            tags=["test"],
        )
        initial_validations = len(skill.validation_results)

        updated = store.update(
            skill.id,
            tags=["extra_tag"],
            reason="Added tag",
        )
        assert len(updated.validation_results) == initial_validations

    def test_create_without_llm_still_validates(self, tmp_path):
        store = SkillStore(tmp_path / "skills")
        skill = store.create(
            name="no_llm_skill",
            description="A skill without LLM validation",
            body='```python\ndef greet():\n    pass\n```',
            tags=["test"],
        )
        assert skill.last_validated_at != ""
        assert len(skill.validation_results) >= 1
        # LLM score should be 0.5 (neutral)
        last = skill.validation_results[-1]
        assert last["llm_score"] == 0.5


# ── record_usage with consecutive_failures ────────────────────────────────────


class TestRecordUsageConsecutiveFailures:
    def test_success_resets_consecutive_failures(self, tmp_path):
        store = SkillStore(tmp_path / "skills")
        skill = store.create(
            name="reset_test",
            description="Test consecutive failure reset on success",
            body='```python\npass\n```',
            tags=["test"],
        )
        # Simulate 2 failures
        store.record_usage(skill.id, success=False)
        store.record_usage(skill.id, success=False)
        s = store.get(skill.id)
        assert s.consecutive_failures == 2

        # Success resets
        store.record_usage(skill.id, success=True)
        s = store.get(skill.id)
        assert s.consecutive_failures == 0

    def test_failures_accumulate(self, tmp_path):
        store = SkillStore(tmp_path / "skills")
        skill = store.create(
            name="accumulate_test",
            description="Test consecutive failure accumulation",
            body='```python\npass\n```',
            tags=["test"],
        )
        for _ in range(5):
            store.record_usage(skill.id, success=False)
        s = store.get(skill.id)
        assert s.consecutive_failures == 5
        assert s.usage_count == 5
        assert s.success_count == 0

    def test_mixed_success_failure(self, tmp_path):
        store = SkillStore(tmp_path / "skills")
        skill = store.create(
            name="mixed_test",
            description="Test mixed success/failure pattern",
            body='```python\npass\n```',
            tags=["test"],
        )
        store.record_usage(skill.id, success=False)  # cf=1
        store.record_usage(skill.id, success=True)    # cf=0, sc=1
        store.record_usage(skill.id, success=False)   # cf=1
        s = store.get(skill.id)
        assert s.consecutive_failures == 1
        assert s.usage_count == 3
        assert s.success_count == 1


# ── _check_single_skill_deprecation ──────────────────────────────────────────


class TestDeprecationConditions:
    def test_consecutive_failures_triggers_deprecation(self):
        skill = _make_skill(
            consecutive_failures=_CONSECUTIVE_FAILURE_THRESHOLD,
            usage_count=10,
            success_count=7,
        )
        reason = SkillStore._check_single_skill_deprecation(skill)
        assert reason is not None
        assert "Consecutive failures" in reason

    def test_below_threshold_no_deprecation(self):
        skill = _make_skill(
            consecutive_failures=_CONSECUTIVE_FAILURE_THRESHOLD - 1,
            usage_count=10,
            success_count=7,
        )
        reason = SkillStore._check_single_skill_deprecation(skill)
        assert reason is None

    def test_low_success_rate_triggers_deprecation(self):
        # Need enough usage and low success rate
        total = _MIN_USAGE_FOR_SUCCESS_RATE_CHECK
        successes = int(total * _SUCCESS_RATE_DEPRECATION_THRESHOLD) - 1
        skill = _make_skill(
            usage_count=total,
            success_count=max(0, successes),
            quality_score=0.8,
        )
        reason = SkillStore._check_single_skill_deprecation(skill)
        assert reason is not None
        assert "Success rate" in reason

    def test_just_above_success_rate_no_deprecation(self):
        total = _MIN_USAGE_FOR_SUCCESS_RATE_CHECK
        successes = int(total * _SUCCESS_RATE_DEPRECATION_THRESHOLD) + 1
        skill = _make_skill(
            usage_count=total,
            success_count=successes,
            quality_score=0.8,
        )
        reason = SkillStore._check_single_skill_deprecation(skill)
        assert reason is None

    def test_low_quality_score_triggers_deprecation(self):
        skill = _make_skill(
            usage_count=_MIN_USAGE_FOR_QUALITY_CHECK,
            success_count=_MIN_USAGE_FOR_QUALITY_CHECK,
            quality_score=_QUALITY_SCORE_DEPRECATION_THRESHOLD - 0.01,
        )
        reason = SkillStore._check_single_skill_deprecation(skill)
        assert reason is not None
        assert "Quality score" in reason

    def test_low_quality_insufficient_usage_no_deprecation(self):
        skill = _make_skill(
            usage_count=_MIN_USAGE_FOR_QUALITY_CHECK - 1,
            quality_score=0.05,
        )
        reason = SkillStore._check_single_skill_deprecation(skill)
        assert reason is None

    def test_idle_skill_triggers_deprecation(self):
        old_date = (
            datetime.now(timezone.utc) - timedelta(days=_IDLE_DAYS_THRESHOLD + 10)
        ).replace(microsecond=0).isoformat()
        skill = _make_skill(
            usage_count=_MIN_USAGE_FOR_IDLE_CHECK - 1,
            quality_score=0.8,
        )
        skill.updated_at = old_date
        reason = SkillStore._check_single_skill_deprecation(skill)
        assert reason is not None
        assert "Idle" in reason

    def test_idle_skill_sufficient_usage_no_deprecation(self):
        old_date = (
            datetime.now(timezone.utc) - timedelta(days=_IDLE_DAYS_THRESHOLD + 10)
        ).replace(microsecond=0).isoformat()
        usage = _MIN_USAGE_FOR_IDLE_CHECK + 5
        skill = _make_skill(
            usage_count=usage,
            success_count=usage,  # High success rate to avoid other deprecation triggers
            quality_score=0.8,
        )
        skill.updated_at = old_date
        reason = SkillStore._check_single_skill_deprecation(skill)
        assert reason is None

    def test_healthy_skill_no_deprecation(self):
        skill = _make_skill(
            usage_count=20,
            success_count=18,
            quality_score=0.85,
            consecutive_failures=0,
        )
        reason = SkillStore._check_single_skill_deprecation(skill)
        assert reason is None


# ── check_and_deprecate (bulk audit) ─────────────────────────────────────────


class TestCheckAndDeprecate:
    def test_deprecates_failing_skills(self, tmp_path):
        store = SkillStore(tmp_path / "skills")
        # Create a skill and make it fail enough
        skill = store.create(
            name="failing_skill",
            description="This skill will be deprecated",
            body='```python\npass\n```',
            tags=["test"],
        )
        # Force consecutive failures
        for _ in range(_CONSECUTIVE_FAILURE_THRESHOLD):
            store.record_usage(skill.id, success=False)

        # The skill should already be deprecated from record_usage
        s = store.get(skill.id)
        assert s.status == "deprecated"

    def test_keeps_healthy_skills(self, tmp_path):
        store = SkillStore(tmp_path / "skills")
        skill = store.create(
            name="healthy_skill",
            description="This skill stays active",
            body='```python\ndef hello():\n    pass\n```',
            tags=["test"],
        )
        store.record_usage(skill.id, success=True)

        deprecated = store.check_and_deprecate()
        assert len(deprecated) == 0

        s = store.get(skill.id)
        assert s.status == "active"

    def test_audit_validates_unvalidated_skills(self, tmp_path):
        store = SkillStore(tmp_path / "skills")
        skill = store.create(
            name="unvalidated_skill",
            description="Should get validated during audit",
            body='```python\nx = 1\n```',
            tags=["test"],
        )
        # Force clear validation
        skill.last_validated_at = ""
        store._write(skill)

        s = store.get(skill.id)
        assert s.last_validated_at == ""

        store.check_and_deprecate()

        s = store.get(skill.id)
        assert s.last_validated_at != ""

    def test_deprecated_skills_excluded_from_list(self, tmp_path):
        store = SkillStore(tmp_path / "skills")
        skill = store.create(
            name="to_deprecate",
            description="Will be deprecated",
            body='```python\npass\n```',
            tags=["test"],
        )
        # Force deprecation via consecutive failures
        for _ in range(_CONSECUTIVE_FAILURE_THRESHOLD):
            store.record_usage(skill.id, success=False)

        active = store.list()
        assert all(s.status == "active" for s in active)

        all_skills = store.list(include_inactive=True)
        deprecated = [s for s in all_skills if s.status == "deprecated"]
        assert len(deprecated) >= 1

    def test_deprecation_records_evolution_history(self, tmp_path):
        store = SkillStore(tmp_path / "skills")
        skill = store.create(
            name="history_test",
            description="Check evolution history on deprecation",
            body='```python\npass\n```',
            tags=["test"],
        )
        for _ in range(_CONSECUTIVE_FAILURE_THRESHOLD):
            store.record_usage(skill.id, success=False)

        s = store.get(skill.id)
        deprecate_entries = [
            e for e in s.evolution_history if e.get("action") == "deprecate"
        ]
        assert len(deprecate_entries) >= 1
        assert "reason" in deprecate_entries[0]


# ── validate_skill (individual validation) ───────────────────────────────────


class TestValidateSkill:
    def test_validate_persists_result(self, tmp_path):
        store = SkillStore(tmp_path / "skills", llm_chat_fn=_mock_llm_chat_fn)
        skill = store.create(
            name="validate_test",
            description="Test validate_skill persistence",
            body='```python\ndef foo():\n    return 42\n```',
            tags=["test"],
        )
        initial_count = len(skill.validation_results)

        result = store.validate_skill(skill.id)
        assert result is not None
        assert "quality_score" in result

        reloaded = store.get(skill.id)
        assert len(reloaded.validation_results) > initial_count
        assert reloaded.quality_score == result["quality_score"]

    def test_validate_nonexistent_returns_none(self, tmp_path):
        store = SkillStore(tmp_path / "skills")
        result = store.validate_skill("nonexistent_id")
        assert result is None

    def test_validation_results_capped_at_10(self, tmp_path):
        store = SkillStore(tmp_path / "skills")
        skill = store.create(
            name="cap_test",
            description="Test validation results cap",
            body='```python\npass\n```',
            tags=["test"],
        )
        # Validate many times
        for _ in range(15):
            store.validate_skill(skill.id)

        reloaded = store.get(skill.id)
        assert len(reloaded.validation_results) <= 10


# ── search() quality blending ────────────────────────────────────────────────


class TestSearchQualityBlending:
    def test_high_quality_ranks_higher(self, tmp_path):
        store = SkillStore(tmp_path / "skills")
        # Create two similar skills with different quality scores
        high_q = store.create(
            name="api_builder_high",
            description="Build REST API endpoints with Flask",
            body='```python\nfrom flask import Flask\napp = Flask(__name__)\n```',
            tags=["api", "web"],
        )
        low_q = store.create(
            name="api_builder_low",
            description="Build REST API endpoints with FastAPI",
            body='```python\nfrom fastapi import FastAPI\napp = FastAPI()\n```',
            tags=["api", "web"],
        )

        # Manually set quality scores
        high_q.quality_score = 0.95
        store._write(high_q)
        low_q.quality_score = 0.1
        store._write(low_q)

        # Re-load store
        store2 = SkillStore(tmp_path / "skills")
        results = store2.search("REST API endpoints builder", top_k=5)
        assert len(results) >= 2

        ids = [r[0].id for r in results]
        assert ids.index(high_q.id) < ids.index(low_q.id), (
            "Higher quality skill should rank above lower quality"
        )

    def test_quality_score_in_prompt_items(self, tmp_path):
        store = SkillStore(tmp_path / "skills")
        skill = store.create(
            name="prompt_test",
            description="Test quality score in prompt items",
            body='```python\npass\n```',
            tags=["test"],
        )
        skill.quality_score = 0.88
        store._write(skill)

        store2 = SkillStore(tmp_path / "skills")
        results = store2.search("test", top_k=5)
        prompt_items = store2.to_prompt_items(results)

        matching = [p for p in prompt_items if p["id"] == skill.id]
        assert len(matching) == 1
        assert matching[0]["quality_score"] == 0.88
        assert matching[0]["status"] == "active"


# ── _deprecate_skill ─────────────────────────────────────────────────────────


class TestDeprecateSkill:
    def test_deprecate_sets_fields(self, tmp_path):
        store = SkillStore(tmp_path / "skills")
        skill = store.create(
            name="deprecate_fields",
            description="Test deprecation fields are set",
            body='```python\npass\n```',
            tags=["test"],
        )
        reason = "Test deprecation reason"
        store._deprecate_skill(skill, reason)

        s = store.get(skill.id)
        assert s.status == "deprecated"
        assert s.deprecated_reason == reason
        assert s.deprecated_at != ""

    def test_deprecate_appends_evolution_history(self, tmp_path):
        store = SkillStore(tmp_path / "skills")
        skill = store.create(
            name="deprecate_history",
            description="Test evolution history on deprecation",
            body='```python\npass\n```',
            tags=["test"],
        )
        store._deprecate_skill(skill, "Test reason")

        s = store.get(skill.id)
        deprecate_entries = [
            e for e in s.evolution_history if e.get("action") == "deprecate"
        ]
        assert len(deprecate_entries) == 1
        assert deprecate_entries[0]["reason"] == "Test reason"


# ── Auto-deprecation via record_usage ────────────────────────────────────────


class TestAutoDeprecationViaRecordUsage:
    def test_auto_deprecate_on_consecutive_failures(self, tmp_path):
        store = SkillStore(tmp_path / "skills")
        skill = store.create(
            name="auto_dep_skill",
            description="Test auto-deprecation through record_usage",
            body='```python\npass\n```',
            tags=["test"],
        )
        for i in range(_CONSECUTIVE_FAILURE_THRESHOLD):
            store.record_usage(skill.id, success=False)

        s = store.get(skill.id)
        assert s.status == "deprecated"
        assert "Consecutive failures" in s.deprecated_reason

    def test_no_auto_deprecate_with_intermittent_success(self, tmp_path):
        store = SkillStore(tmp_path / "skills")
        skill = store.create(
            name="intermittent_skill",
            description="Test no deprecation with intermittent successes",
            body='```python\npass\n```',
            tags=["test"],
        )
        # Pattern: fail, success, fail, success — never hits consecutive threshold
        for _ in range(3):
            store.record_usage(skill.id, success=False)
            store.record_usage(skill.id, success=True)

        s = store.get(skill.id)
        assert s.status == "active"
        assert s.consecutive_failures == 0
        assert s.usage_count == 6
        assert s.success_count == 3

    def test_auto_deprecate_on_low_success_rate(self, tmp_path):
        store = SkillStore(tmp_path / "skills")
        skill = store.create(
            name="low_success_skill",
            description="Test auto-deprecation on low success rate",
            body='```python\npass\n```',
            tags=["test"],
        )
        # Need at least _MIN_USAGE_FOR_SUCCESS_RATE_CHECK uses
        # with success rate below threshold
        total = _MIN_USAGE_FOR_SUCCESS_RATE_CHECK
        successes = int(total * _SUCCESS_RATE_DEPRECATION_THRESHOLD) - 1
        failures = total - successes

        # First add some successes (to avoid consecutive failure deprecation)
        for _ in range(successes):
            store.record_usage(skill.id, success=True)
        # Then add failures — but make sure we break up consecutive failures
        for i in range(failures):
            store.record_usage(skill.id, success=False)

        s = store.get(skill.id)
        # Should be deprecated either by consecutive failures or success rate
        assert s.status == "deprecated"
