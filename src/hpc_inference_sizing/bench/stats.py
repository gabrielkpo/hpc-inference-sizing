"""Per-request timings, step summaries and the service-level objective (protocol §4)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

# At least 15 tokens per second per user: time per output token p95 <= 1/15 s.
TPOT_BOUND_S = 1 / 15


@dataclass
class RequestRecord:
    """Timings of one request, in seconds relative to the start of its load step."""

    sent_at: float
    first_token_at: float | None = None
    last_token_at: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None

    @property
    def ttft(self) -> float | None:
        """Time to first token."""
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.sent_at

    @property
    def tpot(self) -> float | None:
        """Time per output token after the first one."""
        if self.first_token_at is None or self.last_token_at is None:
            return None
        if not self.completion_tokens or self.completion_tokens < 2:
            return None
        return (self.last_token_at - self.first_token_at) / (self.completion_tokens - 1)

    def to_dict(self) -> dict:
        return asdict(self)


def percentile(values: list[float], q: float) -> float:
    """Percentile with linear interpolation between ranks (numpy's default method)."""
    if not values:
        return math.nan
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


@dataclass(frozen=True)
class StepSummary:
    rate: float  # requested arrival rate, requests per second
    duration_s: float
    n_requests: int
    n_errors: int
    ttft_p50: float
    ttft_p95: float
    tpot_p50: float
    tpot_p95: float
    output_tokens_per_s: float

    def meets(self, ttft_target_s: float, tpot_bound_s: float = TPOT_BOUND_S) -> bool:
        """The SLO of protocol §4: TTFT p95 within target, TPOT p95 within bound, no error."""
        if self.n_requests == 0 or self.n_errors > 0:
            return False
        return self.ttft_p95 <= ttft_target_s and self.tpot_p95 <= tpot_bound_s

    def to_dict(self) -> dict:
        return asdict(self)


def summarize(records: list[RequestRecord], rate: float, duration_s: float) -> StepSummary:
    ok = [r for r in records if r.error is None]
    ttfts = [r.ttft for r in ok if r.ttft is not None]
    tpots = [r.tpot for r in ok if r.tpot is not None]
    tokens = sum(r.completion_tokens or 0 for r in ok)
    if ok:
        span = max(r.last_token_at or r.sent_at for r in ok) - min(r.sent_at for r in ok)
    else:
        span = 0.0
    return StepSummary(
        rate=rate,
        duration_s=duration_s,
        n_requests=len(records),
        n_errors=len(records) - len(ok),
        ttft_p50=percentile(ttfts, 50),
        ttft_p95=percentile(ttfts, 95),
        tpot_p50=percentile(tpots, 50),
        tpot_p95=percentile(tpots, 95),
        output_tokens_per_s=tokens / span if span > 0 else 0.0,
    )
