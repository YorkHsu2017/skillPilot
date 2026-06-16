from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class SkillRecord:
    id: str
    name: str
    description: str
    body: str
    tags: list[str] = field(default_factory=list)
    source_task: str = ""
    status: str = "active"
    version: str = "1.0.0"
    usage_count: int = 0
    success_count: int = 0
    quality_score: float = 0.5
    validation_results: list[dict[str, Any]] = field(default_factory=list)
    last_validated_at: str = ""
    consecutive_failures: int = 0
    deprecated_at: str = ""
    deprecated_reason: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    evolution_history: list[dict[str, Any]] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillRecord":
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            description=str(data.get("description", "")),
            body=str(data.get("body", "")),
            tags=list(data.get("tags", [])),
            source_task=str(data.get("source_task", "")),
            status=str(data.get("status", "active")),
            version=str(data.get("version", "1.0.0")),
            usage_count=int(data.get("usage_count", 0)),
            success_count=int(data.get("success_count", 0)),
            quality_score=float(data.get("quality_score", 0.5)),
            validation_results=list(data.get("validation_results", [])),
            last_validated_at=str(data.get("last_validated_at", "")),
            consecutive_failures=int(data.get("consecutive_failures", 0)),
            deprecated_at=str(data.get("deprecated_at", "")),
            deprecated_reason=str(data.get("deprecated_reason", "")),
            created_at=str(data.get("created_at", utc_now())),
            updated_at=str(data.get("updated_at", utc_now())),
            evolution_history=list(data.get("evolution_history", [])),
            embedding=list(data.get("embedding", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def bump_patch(self) -> None:
        parts = self.version.split(".")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            self.version = "1.0.1"
            return
        major, minor, patch = (int(part) for part in parts)
        self.version = f"{major}.{minor}.{patch + 1}"


@dataclass
class ToolCallLog:
    name: str
    args: dict[str, Any]
    result: dict[str, Any]
    success: bool


@dataclass
class AgentResult:
    success: bool
    final: str
    tool_calls: list[ToolCallLog]
    selected_skill_ids: list[str] = field(default_factory=list)
    lifecycle_action: str = "none"
    lifecycle_skill_id: str | None = None
    plan: list[str] | None = None
    summary: str | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
