from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Optional


class SafetyLeaseFault(RuntimeError):
    pass


@dataclass(frozen=True)
class SafetyHeartbeat:
    source_pid: int
    source_host: str
    session_id: str
    sequence: int
    sent_monotonic: float
    safety_state: str
    estop_latched: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> "SafetyHeartbeat":
        try:
            data = json.loads(payload)
            result = cls(
                source_pid=int(data["source_pid"]),
                source_host=str(data["source_host"]),
                session_id=str(data["session_id"]),
                sequence=int(data["sequence"]),
                sent_monotonic=float(data["sent_monotonic"]),
                safety_state=str(data["safety_state"]),
                estop_latched=bool(data["estop_latched"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SafetyLeaseFault(f"invalid safety heartbeat: {exc}") from exc
        if (
            result.source_pid <= 0
            or not result.source_host
            or not result.session_id
            or result.sequence <= 0
            or not math.isfinite(result.sent_monotonic)
        ):
            raise SafetyLeaseFault("safety heartbeat has invalid identity/sequence/time")
        return result


class SafetyLeaseMonitor:
    def __init__(self, *, timeout_sec: float) -> None:
        self.timeout_sec = float(timeout_sec)
        if not math.isfinite(self.timeout_sec) or self.timeout_sec <= 0.0:
            raise ValueError("safety lease timeout must be positive")
        self.session_id: Optional[str] = None
        self.source_pid: Optional[int] = None
        self.source_host: Optional[str] = None
        self.sequence = 0
        self.last_receive_monotonic = -1.0

    @property
    def has_session(self) -> bool:
        return self.session_id is not None

    def observe(self, heartbeat: SafetyHeartbeat, *, received_at: float) -> None:
        received_at = float(received_at)
        if not math.isfinite(received_at):
            raise SafetyLeaseFault("safety heartbeat receive time is invalid")
        if self.session_id is None:
            self.session_id = heartbeat.session_id
            self.source_pid = heartbeat.source_pid
            self.source_host = heartbeat.source_host
        elif (
            heartbeat.session_id != self.session_id
            or heartbeat.source_pid != self.source_pid
            or heartbeat.source_host != self.source_host
        ):
            raise SafetyLeaseFault(
                "safety publisher identity/session changed; explicit operator recovery required"
            )
        if heartbeat.sequence <= self.sequence:
            raise SafetyLeaseFault(
                f"safety heartbeat sequence {heartbeat.sequence} is not newer than {self.sequence}"
            )
        self.sequence = heartbeat.sequence
        self.last_receive_monotonic = received_at

    def is_healthy(self, *, now: float) -> bool:
        now = float(now)
        return (
            self.has_session
            and math.isfinite(now)
            and self.last_receive_monotonic >= 0.0
            and 0.0 <= now - self.last_receive_monotonic <= self.timeout_sec
        )
