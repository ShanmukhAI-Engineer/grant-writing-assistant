from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json


@dataclass
class Grant:
    id: str
    title: str
    source: str
    link: str
    amount_min: int | None
    amount_max: int | None
    deadline: str | None  # ISO date string
    geographies: List[str]
    org_types: List[str]
    focus_areas: List[str]
    eligibility_notes: str | None = None

    def deadline_days_from_now(self) -> int | None:
        if not self.deadline:
            return None
        try:
            d = datetime.fromisoformat(self.deadline)
        except Exception:
            return None
        return (d - datetime.now()).days


def load_local_grants(path: Path) -> List[Grant]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Grant(**g) for g in data]


@dataclass
class Answers:
    org_type: str
    funding_min: int
    funding_max: int
    focus_areas: List[str]
    geographies: List[str]
    deadline_within_days: int | None = 90


@dataclass
class ScoreBreakdown:
    org_type: float
    funding: float
    focus: float
    geo: float
    deadline: float

    def total(self, weights: Dict[str, float]) -> float:
        return (
            self.org_type * weights.get("org_type", 0)
            + self.funding * weights.get("funding", 0)
            + self.focus * weights.get("focus", 0)
            + self.geo * weights.get("geo", 0)
            + self.deadline * weights.get("deadline", 0)
        )


DEFAULT_WEIGHTS: Dict[str, float] = {
    "org_type": 0.40,
    "funding": 0.30,
    "focus": 0.15,
    "geo": 0.10,
    "deadline": 0.05,
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _range_overlap_score(desired: Tuple[int, int], offered: Tuple[int | None, int | None]) -> float:
    dmin, dmax = desired
    omin, omax = offered
    # Handle unknown offered ranges conservatively
    if omin is None and omax is None:
        return 0.0
    if omin is None:
        omin = 0
    if omax is None:
        omax = max(dmax, omin)
    if dmax <= dmin:
        return 0.0
    # Overlap length / desired length
    left = max(dmin, omin)
    right = min(dmax, omax)
    overlap = max(0, right - left)
    return _clamp01(overlap / float(dmax - dmin))


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(map(str.lower, a)), set(map(str.lower, b))
    if not sa and not sb:
        return 0.0
    return _clamp01(len(sa & sb) / float(len(sa | sb)))


def score_grant(grant: Grant, answers: Answers, *, weights: Dict[str, float] | None = None) -> Tuple[float, ScoreBreakdown]:
    weights = weights or DEFAULT_WEIGHTS

    # Org type: exact inclusion
    org_score = 1.0 if answers.org_type.lower() in {t.lower() for t in grant.org_types} else 0.0

    # Funding: overlap of desired range with grant range
    fund_score = _range_overlap_score((answers.funding_min, answers.funding_max), (grant.amount_min, grant.amount_max))

    # Focus: jaccard overlap
    focus_score = _jaccard(answers.focus_areas, grant.focus_areas)

    # Geography: treat missing/Global as wildcard
    if not grant.geographies or any(g.lower() in {"any", "global"} for g in grant.geographies):
        geo_score = 1.0
    else:
        geo_score = _jaccard(answers.geographies, grant.geographies)

    # Deadline: if provided, closer deadlines inside window score higher
    deadline_score = 0.0
    if answers.deadline_within_days and answers.deadline_within_days > 0:
        days = grant.deadline_days_from_now()
        if days is not None and days >= 0 and days <= answers.deadline_within_days:
            deadline_score = _clamp01(1.0 - (days / float(answers.deadline_within_days)))

    breakdown = ScoreBreakdown(org_type=org_score, funding=fund_score, focus=focus_score, geo=geo_score, deadline=deadline_score)
    total = breakdown.total(weights)
    return total, breakdown


def rank_grants(grants: List[Grant], answers: Answers, *, top_n: int = 10, weights: Dict[str, float] | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for g in grants:
        total, br = score_grant(g, answers, weights=weights)
        rows.append({
            "id": g.id,
            "title": g.title,
            "source": g.source,
            "link": g.link,
            "amount_min": g.amount_min,
            "amount_max": g.amount_max,
            "deadline": g.deadline,
            "geographies": g.geographies,
            "org_types": g.org_types,
            "focus_areas": g.focus_areas,
            "score": round(total, 4),
            "score_breakdown": {
                "org_type": round(br.org_type, 4),
                "funding": round(br.funding, 4),
                "focus": round(br.focus, 4),
                "geo": round(br.geo, 4),
                "deadline": round(br.deadline, 4),
            }
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:top_n]
