from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, List

from .models import SkillRecord, utc_now


EmbedFn = Callable[[List[str]], List[List[float]]]
LLMChatFn = Callable[..., str]  # (messages, temperature, max_tokens) -> str


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "into",
    "your",
    "you",
    "are",
    "can",
    "will",
    "should",
    "一个",
    "任务",
    "实现",
    "生成",
    "代码",
}


def _slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "skill"


def _tokens(text: str) -> set[str]:
    lowered = text.lower()
    result = set(re.findall(r"[a-z0-9_]{2,}", lowered))
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    result.update(chinese_chars)
    chinese_text = "".join(chinese_chars)
    for index in range(len(chinese_text) - 1):
        result.add(chinese_text[index : index + 2])
    return {token for token in result if token not in _STOPWORDS}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors using pure Python."""
    if len(a) != len(b) or not a:
        return 0.0
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def _skill_text(skill: SkillRecord) -> str:
    """Compose text representation of a skill for embedding."""
    parts = [
        skill.name,
        skill.description,
        skill.body[:2000],  # Limit body length to control embedding cost
        " ".join(skill.tags),
        skill.source_task,
    ]
    return " ".join(part for part in parts if part)


# ── Quality control thresholds ────────────────────────────────────────────────
_CONSECUTIVE_FAILURE_THRESHOLD = 3       # Deprecate after N consecutive failures
_MIN_USAGE_FOR_SUCCESS_RATE_CHECK = 5    # Need at least N uses before checking success rate
_SUCCESS_RATE_DEPRECATION_THRESHOLD = 0.3  # Deprecate if success rate below this
_QUALITY_SCORE_DEPRECATION_THRESHOLD = 0.2  # Deprecate if quality below this
_MIN_USAGE_FOR_QUALITY_CHECK = 3         # Need at least N uses before quality-based deprecation
_IDLE_DAYS_THRESHOLD = 90                # Deprecate if unused for N days
_MIN_USAGE_FOR_IDLE_CHECK = 2            # Only deprecate idle skills with very few uses


class SkillValidator:
    """Validates skill quality across three dimensions: static, syntax, and LLM."""

    @staticmethod
    def static_check(skill: SkillRecord) -> float:
        """Static analysis: body length, description non-empty, tags, code presence.
        Returns a score in [0.0, 1.0].
        """
        score = 0.0
        checks = 0

        # Check 1: body length >= 20 chars (weight 0.3)
        checks += 1
        if len(skill.body) >= 20:
            score += 0.3
        else:
            score += 0.3 * min(len(skill.body), 20) / 20

        # Check 2: description non-empty and >= 10 chars (weight 0.3)
        checks += 1
        if len(skill.description.strip()) >= 10:
            score += 0.3
        elif skill.description.strip():
            score += 0.3 * len(skill.description.strip()) / 10

        # Check 3: has at least one tag (weight 0.2)
        checks += 1
        if skill.tags:
            score += 0.2

        # Check 4: body contains code-like content (indentation, keywords, parens) (weight 0.2)
        checks += 1
        code_indicators = [
            r"\bdef\b", r"\bclass\b", r"\bimport\b", r"\bif\b.*:",
            r"\bfor\b.*:", r"\breturn\b", r"\bprint\b",
            r"\{.*\}", r"\(.*\)", r"=\s",
            r"^\s{2,}",  # indentation
        ]
        code_matches = sum(
            1 for pattern in code_indicators
            if re.search(pattern, skill.body, re.MULTILINE)
        )
        score += 0.2 * min(1.0, code_matches / 3)

        return min(1.0, score)

    @staticmethod
    def syntax_check(skill: SkillRecord) -> float:
        """Extract Python code blocks from body and validate syntax with ast.parse.
        Returns a score in [0.0, 1.0].
        """
        code_blocks = re.findall(
            r"```(?:python|py)?\s*\n(.*?)```",
            skill.body,
            re.DOTALL,
        )
        # Also look for indented Python code (4+ spaces, common in skill bodies)
        if not code_blocks:
            # Try to find lines that look like Python code
            lines = skill.body.splitlines()
            python_lines = [
                line for line in lines
                if re.match(r"^\s{2,}(def |class |import |from |if |for |while |return |print )", line)
            ]
            if python_lines:
                # Try to parse the whole body as Python
                code_blocks = [skill.body]

        if not code_blocks:
            # No code to validate — return neutral score
            return 0.5

        valid = 0
        total = len(code_blocks)
        for block in code_blocks:
            try:
                ast.parse(block.strip())
                valid += 1
            except SyntaxError:
                pass

        return valid / total if total > 0 else 0.5

    @staticmethod
    def llm_quality_check(skill: SkillRecord, llm_chat_fn: LLMChatFn | None) -> float:
        """Ask the LLM to rate the skill on clarity, reusability, completeness (1-5).
        Returns a normalized score in [0.0, 1.0].
        """
        if llm_chat_fn is None:
            return 0.5  # Neutral if no LLM available

        prompt = {
            "skill_name": skill.name,
            "skill_description": skill.description,
            "skill_body": skill.body[:2000],
            "tags": skill.tags,
            "instruction": (
                "Rate this skill on three dimensions (1-5 scale each):\n"
                "1. Clarity: Is the description and body easy to understand?\n"
                "2. Reusability: Is this pattern general enough to be reused across tasks?\n"
                "3. Completeness: Does the body contain enough detail (code, steps) to follow?\n"
                "Return strict JSON: {\"clarity\": <1-5>, \"reusability\": <1-5>, \"completeness\": <1-5>}"
            ),
        }
        try:
            raw = llm_chat_fn(
                [
                    {
                        "role": "system",
                        "content": "You evaluate reusable coding skill quality. Return strict JSON only.",
                    },
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                0.0,
                300,
            )
        except (RuntimeError, Exception):
            return 0.5  # Neutral on failure

        try:
            # Parse JSON from response
            stripped = raw.strip()
            if stripped.startswith("```"):
                lines = stripped.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                stripped = "\n".join(lines).strip()
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                return 0.5
            clarity = float(parsed.get("clarity", 3))
            reusability = float(parsed.get("reusability", 3))
            completeness = float(parsed.get("completeness", 3))
            # Clamp to [1, 5]
            clarity = max(1.0, min(5.0, clarity))
            reusability = max(1.0, min(5.0, reusability))
            completeness = max(1.0, min(5.0, completeness))
            avg = (clarity + reusability + completeness) / 3.0
            return (avg - 1.0) / 4.0  # Normalize [1,5] -> [0.0, 1.0]
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            return 0.5

    @classmethod
    def validate(
        cls,
        skill: SkillRecord,
        *,
        llm_chat_fn: LLMChatFn | None = None,
    ) -> dict[str, Any]:
        """Run all validation dimensions and return a result dict.
        
        Weights: static 30%, syntax 30%, LLM 40%.
        """
        static = cls.static_check(skill)
        syntax = cls.syntax_check(skill)
        llm = cls.llm_quality_check(skill, llm_chat_fn)

        quality = static * 0.3 + syntax * 0.3 + llm * 0.4
        quality = round(max(0.0, min(1.0, quality)), 4)

        return {
            "timestamp": utc_now(),
            "quality_score": quality,
            "static_score": round(static, 4),
            "syntax_score": round(syntax, 4),
            "llm_score": round(llm, 4),
        }


class SkillStore:
    """Local skill store compatible with the EvoTool idea: one folder per skill."""

    def __init__(
        self,
        skills_dir: str | Path,
        *,
        embed_fn: EmbedFn | None = None,
        llm_chat_fn: LLMChatFn | None = None,
    ) -> None:
        self.skills_dir = Path(skills_dir).expanduser().resolve()
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.embed_fn = embed_fn
        self.llm_chat_fn = llm_chat_fn

    def _skill_dir(self, skill_id: str) -> Path:
        return self.skills_dir / skill_id

    def _skill_path(self, skill_id: str) -> Path:
        return self._skill_dir(skill_id) / "skill.json"

    def _write(self, skill: SkillRecord) -> None:
        directory = self._skill_dir(skill.id)
        directory.mkdir(parents=True, exist_ok=True)
        path = self._skill_path(skill.id)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(skill.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def list(self, *, include_inactive: bool = False) -> list[SkillRecord]:
        records: list[SkillRecord] = []
        for path in sorted(self.skills_dir.glob("*/skill.json")):
            try:
                record = SkillRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if include_inactive or record.status == "active":
                records.append(record)
        return sorted(records, key=lambda item: item.updated_at, reverse=True)

    def get(self, skill_id: str) -> SkillRecord | None:
        path = self._skill_path(skill_id)
        if not path.exists():
            return None
        try:
            return SkillRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def search(self, query: str, *, top_k: int = 5) -> list[tuple[SkillRecord, float]]:
        all_skills = self.list()
        if not all_skills:
            return []

        query_tokens = _tokens(query)
        query_lower = query.lower()

        # Try semantic retrieval if embed_fn is available
        query_embedding: list[float] | None = None
        if self.embed_fn is not None:
            try:
                embeddings = self.embed_fn([query])
                if embeddings and len(embeddings[0]) > 0:
                    query_embedding = embeddings[0]
            except (RuntimeError, Exception):
                query_embedding = None

        scored: list[tuple[SkillRecord, float]] = []
        for skill in all_skills:
            # Lexical score (always computed as fallback / blending component)
            lexical_score = self._lexical_score(skill, query_tokens, query_lower)

            # Semantic score
            semantic_score = 0.0
            if query_embedding is not None and skill.embedding:
                semantic_score = max(0.0, _cosine_similarity(query_embedding, skill.embedding))

            # Quality score component
            quality = skill.quality_score

            # Success rate component
            success_rate_score = 0.0
            if skill.usage_count >= 3:
                success_rate_score = skill.success_count / skill.usage_count

            # Blend scores
            if query_embedding is not None and skill.embedding:
                # Hybrid: 50% semantic + 25% lexical + 15% quality + 10% success rate
                score = (
                    0.50 * semantic_score
                    + 0.25 * lexical_score
                    + 0.15 * quality
                    + 0.10 * success_rate_score
                )
            elif query_embedding is not None and not skill.embedding:
                # Skill has no embedding yet: lexical + quality + success
                score = 0.60 * lexical_score + 0.25 * quality + 0.15 * success_rate_score
            else:
                # No embed_fn available: lexical + quality + success
                score = 0.60 * lexical_score + 0.25 * quality + 0.15 * success_rate_score

            score = round(min(1.0, score), 4)
            if score > 0.05:
                scored.append((skill, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _lexical_score(
        skill: SkillRecord,
        query_tokens: set[str],
        query_lower: str,
    ) -> float:
        haystack = " ".join(
            [
                skill.name,
                skill.description,
                skill.body,
                skill.source_task,
                " ".join(skill.tags),
            ]
        )
        haystack_lower = haystack.lower()
        haystack_tokens = _tokens(haystack)
        token_score = len(query_tokens & haystack_tokens) / max(len(query_tokens), 1)
        phrase_bonus = 0.25 if query_lower and query_lower in haystack_lower else 0.0
        fuzzy_score = SequenceMatcher(None, query_lower[:300], haystack_lower[:1000]).ratio() * 0.25
        return min(1.0, token_score + phrase_bonus + fuzzy_score)

    def create(
        self,
        *,
        name: str,
        description: str,
        body: str,
        tags: list[str] | None = None,
        source_task: str = "",
    ) -> SkillRecord:
        base_id = _slugify(name)
        suffix = hashlib.sha1(f"{name}:{source_task}:{body}".encode("utf-8")).hexdigest()[:8]
        skill_id = f"{base_id}_{suffix}"
        skill = SkillRecord(
            id=skill_id,
            name=name.strip() or base_id,
            description=description.strip(),
            body=body.strip(),
            tags=tags or [],
            source_task=source_task,
        )
        # Generate embedding if embed_fn is available
        if self.embed_fn is not None:
            try:
                text = _skill_text(skill)
                embeddings = self.embed_fn([text])
                if embeddings and len(embeddings[0]) > 0:
                    skill.embedding = embeddings[0]
            except (RuntimeError, Exception):
                pass  # Silently skip if embedding fails
        skill.evolution_history.append({"timestamp": utc_now(), "action": "create"})
        # Auto-validate on creation
        self._auto_validate(skill)
        self._write(skill)
        return skill

    def update(
        self,
        skill_id: str,
        *,
        description: str | None = None,
        body: str | None = None,
        tags: list[str] | None = None,
        reason: str = "",
    ) -> SkillRecord:
        skill = self.get(skill_id)
        if skill is None:
            raise ValueError(f"Skill not found: {skill_id}")

        changed: list[str] = []
        if description is not None and description.strip() and description.strip() != skill.description:
            skill.description = description.strip()
            changed.append("description")
        if body is not None and body.strip() and body.strip() != skill.body:
            skill.body = body.strip()
            changed.append("body")
        if tags is not None:
            merged_tags = sorted(set(skill.tags) | {tag for tag in tags if tag})
            if merged_tags != skill.tags:
                skill.tags = merged_tags
                changed.append("tags")
        if changed:
            # Regenerate embedding if content changed and embed_fn is available
            if self.embed_fn is not None and any(field in changed for field in ("description", "body", "tags")):
                try:
                    text = _skill_text(skill)
                    embeddings = self.embed_fn([text])
                    if embeddings and len(embeddings[0]) > 0:
                        skill.embedding = embeddings[0]
                except (RuntimeError, Exception):
                    pass  # Keep old embedding if regeneration fails
            old_version = skill.version
            skill.bump_patch()
            skill.updated_at = utc_now()
            skill.evolution_history.append(
                {
                    "timestamp": skill.updated_at,
                    "action": "update",
                    "changed_fields": changed,
                    "old_version": old_version,
                    "new_version": skill.version,
                    "reason": reason,
                }
            )
            # Auto-validate on update (body or description changed)
            if any(field in changed for field in ("description", "body")):
                self._auto_validate(skill)
            self._write(skill)
        return skill

    def record_usage(self, skill_id: str, *, success: bool) -> None:
        skill = self.get(skill_id)
        if skill is None:
            return
        skill.usage_count += 1
        if success:
            skill.success_count += 1
            skill.consecutive_failures = 0  # Reset on success
        else:
            skill.consecutive_failures += 1
        skill.updated_at = utc_now()
        # Check deprecation after usage recording
        deprecation_reason = self._check_single_skill_deprecation(skill)
        if deprecation_reason:
            self._deprecate_skill(skill, deprecation_reason)
        self._write(skill)

    def to_prompt_items(self, scored: list[tuple[SkillRecord, float]]) -> list[dict[str, Any]]:
        return [
            {
                "id": skill.id,
                "name": skill.name,
                "description": skill.description,
                "tags": skill.tags,
                "score": score,
                "body": skill.body[:4000],
                "usage_count": skill.usage_count,
                "success_count": skill.success_count,
                "quality_score": round(skill.quality_score, 2),
                "status": skill.status,
            }
            for skill, score in scored
        ]

    def validate_skill(self, skill_id: str) -> dict[str, Any] | None:
        """Run full validation on a single skill and persist the result.
        Returns the validation result dict, or None if skill not found.
        """
        skill = self.get(skill_id)
        if skill is None:
            return None
        result = SkillValidator.validate(skill, llm_chat_fn=self.llm_chat_fn)
        skill.quality_score = result["quality_score"]
        skill.last_validated_at = result["timestamp"]
        skill.validation_results.append(result)
        # Keep only the last 10 validation results
        if len(skill.validation_results) > 10:
            skill.validation_results = skill.validation_results[-10:]
        skill.updated_at = utc_now()
        skill.evolution_history.append({
            "timestamp": utc_now(),
            "action": "validate",
            "quality_score": result["quality_score"],
        })
        self._write(skill)
        return result

    def check_and_deprecate(self) -> list[dict[str, str]]:
        """Audit all active skills and deprecate those that fail quality checks.
        Returns a list of dicts describing each deprecation action.
        """
        deprecated: list[dict[str, str]] = []
        for skill in self.list():  # Only active skills
            # Re-validate if not validated yet or validation is stale
            if not skill.last_validated_at:
                self.validate_skill(skill.id)
                skill = self.get(skill.id)  # Reload after validation
                if skill is None:
                    continue

            reason = self._check_single_skill_deprecation(skill)
            if reason:
                self._deprecate_skill(skill, reason)
                deprecated.append({
                    "skill_id": skill.id,
                    "name": skill.name,
                    "reason": reason,
                })
        return deprecated

    def _auto_validate(self, skill: SkillRecord) -> None:
        """Run validation and update quality_score in-place (does NOT persist)."""
        result = SkillValidator.validate(skill, llm_chat_fn=self.llm_chat_fn)
        skill.quality_score = result["quality_score"]
        skill.last_validated_at = result["timestamp"]
        skill.validation_results.append(result)
        if len(skill.validation_results) > 10:
            skill.validation_results = skill.validation_results[-10:]
        skill.evolution_history.append({
            "timestamp": utc_now(),
            "action": "validate",
            "quality_score": result["quality_score"],
        })

    @staticmethod
    def _check_single_skill_deprecation(skill: SkillRecord) -> str | None:
        """Check whether a single skill should be deprecated.
        Returns a reason string, or None if the skill is still healthy.
        """
        # Condition 1: consecutive failures
        if skill.consecutive_failures >= _CONSECUTIVE_FAILURE_THRESHOLD:
            return (
                f"Consecutive failures reached {skill.consecutive_failures} "
                f"(threshold: {_CONSECUTIVE_FAILURE_THRESHOLD})"
            )

        # Condition 2: low success rate with sufficient usage
        if skill.usage_count >= _MIN_USAGE_FOR_SUCCESS_RATE_CHECK:
            success_rate = skill.success_count / skill.usage_count
            if success_rate < _SUCCESS_RATE_DEPRECATION_THRESHOLD:
                return (
                    f"Success rate {success_rate:.1%} below threshold "
                    f"{_SUCCESS_RATE_DEPRECATION_THRESHOLD:.0%} "
                    f"after {skill.usage_count} uses"
                )

        # Condition 3: low quality score with sufficient usage
        if skill.usage_count >= _MIN_USAGE_FOR_QUALITY_CHECK:
            if skill.quality_score < _QUALITY_SCORE_DEPRECATION_THRESHOLD:
                return (
                    f"Quality score {skill.quality_score:.2f} below threshold "
                    f"{_QUALITY_SCORE_DEPRECATION_THRESHOLD} "
                    f"after {skill.usage_count} uses"
                )

        # Condition 4: idle for too long with very few uses
        if skill.usage_count < _MIN_USAGE_FOR_IDLE_CHECK:
            try:
                last_used = datetime.fromisoformat(skill.updated_at)
                if last_used.tzinfo is None:
                    last_used = last_used.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                idle_days = (now - last_used).days
                if idle_days >= _IDLE_DAYS_THRESHOLD:
                    return (
                        f"Idle for {idle_days} days (threshold: {_IDLE_DAYS_THRESHOLD}) "
                        f"with only {skill.usage_count} uses"
                    )
            except (ValueError, TypeError):
                pass

        return None

    def _deprecate_skill(self, skill: SkillRecord, reason: str) -> None:
        """Mark a skill as deprecated and persist the change."""
        skill.status = "deprecated"
        skill.deprecated_at = utc_now()
        skill.deprecated_reason = reason
        skill.evolution_history.append({
            "timestamp": utc_now(),
            "action": "deprecate",
            "reason": reason,
        })
        self._write(skill)
