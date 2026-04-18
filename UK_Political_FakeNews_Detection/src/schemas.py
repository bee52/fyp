from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_text: str = Field(min_length=1)
    stack: Literal["sklearn", "roberta"] = "sklearn"

    @field_validator("raw_text")
    @classmethod
    def strip_and_validate_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("raw_text cannot be empty")
        return cleaned


class BranchScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    style_fake_probability: float = Field(ge=0.0, le=1.0)
    semantic_fake_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    fusion_fake_probability: float = Field(ge=0.0, le=1.0)


class StylisticBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    word_count: float = Field(ge=0.0)
    shout_ratio: float = Field(ge=0.0, le=1.0)
    exclamation_density: float = Field(ge=0.0)
    question_density: float = Field(ge=0.0)
    lexical_diversity: float = Field(ge=0.0)
    sentiment: float = Field(ge=-1.0, le=1.0)


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: Literal["Reliable", "Unreliable"]
    confidence: float = Field(ge=0.0, le=1.0)
    fake_probability: float = Field(ge=0.0, le=1.0)
    stack: Literal["sklearn", "roberta"]
    branch_scores: BranchScores
    stylistic_breakdown: StylisticBreakdown
