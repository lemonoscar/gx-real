from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Mapping, Optional, Tuple


JOYSTICK_AXES = ("lx", "ly", "rx", "ry")


def handover_allows_motion(policy_elapsed: float, handover_duration: float) -> bool:
    elapsed = _nonnegative_float(policy_elapsed, "policy_elapsed")
    duration = _nonnegative_float(handover_duration, "handover_duration")
    return elapsed >= duration


@dataclass(frozen=True)
class BaseCommand:
    vx: float
    vy: float
    yaw_rate: float
    stamp: float
    source: str
    valid: bool = True
    inhibited: bool = False
    reason: str = ""

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.vx, self.vy, self.yaw_rate)


@dataclass(frozen=True)
class BaseCommandGate:
    standup_done: bool
    policy_running: bool
    lowlevel_align_done: bool
    emergency_stop: bool = False

    @property
    def allows_motion(self) -> bool:
        return (
            self.standup_done
            and self.policy_running
            and self.lowlevel_align_done
            and not self.emergency_stop
        )


class FixedCommandProvider:
    def __init__(self, vx: float, vy: float, yaw_rate: float):
        self.command = (
            _finite_float(vx, "cmd_vx"),
            _finite_float(vy, "cmd_vy"),
            _finite_float(yaw_rate, "cmd_yaw"),
        )

    def update(self, now: Optional[float] = None) -> BaseCommand:
        stamp = time.monotonic() if now is None else float(now)
        return BaseCommand(*self.command, stamp=stamp, source="fixed", valid=True)


class WirelessJoystickCommandProvider:
    def __init__(
        self,
        *,
        vx_axis: str = "ly",
        vx_sign: int = 1,
        vy_axis: str = "lx",
        vy_sign: int = -1,
        yaw_axis: str = "rx",
        yaw_sign: int = -1,
        deadzone: float = 0.12,
        min_vx: float = 0.20,
        max_vx: float = 0.50,
        max_vy: float = 0.20,
        max_yaw: float = 0.50,
        watchdog_sec: float = 0.25,
    ):
        self.vx_axis = _validate_axis(vx_axis, "vx_axis")
        self.vx_sign = _validate_sign(vx_sign, "vx_sign")
        self.vy_axis = _validate_axis(vy_axis, "vy_axis")
        self.vy_sign = _validate_sign(vy_sign, "vy_sign")
        self.yaw_axis = _validate_axis(yaw_axis, "yaw_axis")
        self.yaw_sign = _validate_sign(yaw_sign, "yaw_sign")
        self.deadzone = _nonnegative_float(deadzone, "deadzone")
        if self.deadzone >= 1.0:
            raise ValueError(f"deadzone must be < 1, got {self.deadzone!r}")
        self.min_vx = _nonnegative_float(min_vx, "min_vx")
        self.max_vx = _nonnegative_float(max_vx, "max_vx")
        if self.min_vx > self.max_vx:
            raise ValueError(
                f"min_vx must be <= max_vx, got {self.min_vx!r} > {self.max_vx!r}"
            )
        self.max_vy = _nonnegative_float(max_vy, "max_vy")
        self.max_yaw = _nonnegative_float(max_yaw, "max_yaw")
        self.watchdog_sec = _positive_float(watchdog_sec, "watchdog_sec")
        self.axes = {axis: 0.0 for axis in JOYSTICK_AXES}
        self.last_msg_time = -1.0

    def update_wireless(
        self,
        *,
        lx: float,
        ly: float,
        rx: float,
        ry: float,
        stamp: Optional[float] = None,
    ) -> None:
        self.axes = {
            "lx": _clip_axis(lx),
            "ly": _clip_axis(ly),
            "rx": _clip_axis(rx),
            "ry": _clip_axis(ry),
        }
        self.last_msg_time = time.monotonic() if stamp is None else float(stamp)

    def update_message(self, msg, stamp: Optional[float] = None) -> None:
        self.update_wireless(
            lx=float(getattr(msg, "lx")),
            ly=float(getattr(msg, "ly")),
            rx=float(getattr(msg, "rx")),
            ry=float(getattr(msg, "ry")),
            stamp=stamp,
        )

    def axes_centered(self) -> bool:
        return all(abs(value) <= self.deadzone for value in self.axes.values())

    def update(self, now: Optional[float] = None) -> BaseCommand:
        stamp = time.monotonic() if now is None else float(now)
        if self.last_msg_time < 0.0:
            return BaseCommand(
                0.0,
                0.0,
                0.0,
                stamp=stamp,
                source="wireless_joystick",
                valid=False,
                reason="wirelesscontroller_missing",
            )
        age = stamp - self.last_msg_time
        if age > self.watchdog_sec:
            return BaseCommand(
                0.0,
                0.0,
                0.0,
                stamp=stamp,
                source="wireless_joystick",
                valid=False,
                reason="wirelesscontroller_stale",
            )

        vx = self.vx_sign * self._axis_speed(
            self.vx_axis,
            self.min_vx,
            self.max_vx,
        )
        vy = self.vy_sign * self._axis(self.vy_axis) * self.max_vy
        yaw_rate = self.yaw_sign * self._axis(self.yaw_axis) * self.max_yaw
        return BaseCommand(
            vx,
            vy,
            yaw_rate,
            stamp=stamp,
            source="wireless_joystick",
            valid=True,
        )

    def _axis(self, axis_name: str) -> float:
        value = self.axes[axis_name]
        if abs(value) <= self.deadzone:
            return 0.0
        return value

    def _axis_speed(
        self,
        axis_name: str,
        min_speed: float,
        max_speed: float,
    ) -> float:
        value = self.axes[axis_name]
        magnitude = abs(value)
        if magnitude <= self.deadzone or max_speed == 0.0:
            return 0.0
        active_ratio = (magnitude - self.deadzone) / (1.0 - self.deadzone)
        speed = min_speed + active_ratio * (max_speed - min_speed)
        return math.copysign(speed, value)


class CommandSafetyFilter:
    def __init__(
        self,
        *,
        acc_vx: float = 0.3,
        acc_vy: float = 0.3,
        acc_yaw: float = 0.6,
        dry_run: bool = False,
    ):
        self.acc_limits = (
            _nonnegative_float(acc_vx, "acc_vx"),
            _nonnegative_float(acc_vy, "acc_vy"),
            _nonnegative_float(acc_yaw, "acc_yaw"),
        )
        self.dry_run = bool(dry_run)
        self.last_command = (0.0, 0.0, 0.0)
        self.last_update_time = -1.0
        self.joystick_inhibited = False

    def inhibit_until_centered(self) -> None:
        self.joystick_inhibited = True

    def reset(self, command: Tuple[float, float, float] = (0.0, 0.0, 0.0), now: Optional[float] = None) -> None:
        self.last_command = tuple(float(value) for value in command)
        self.last_update_time = time.monotonic() if now is None else float(now)

    def update(
        self,
        raw_command: BaseCommand,
        gate: BaseCommandGate,
        *,
        axes_centered: bool = True,
        now: Optional[float] = None,
    ) -> BaseCommand:
        stamp = time.monotonic() if now is None else float(now)
        if self.joystick_inhibited and axes_centered:
            self.joystick_inhibited = False

        reason = raw_command.reason
        valid = raw_command.valid
        desired = raw_command.as_tuple()

        if self.dry_run:
            desired = (0.0, 0.0, 0.0)
            reason = "dry_run"
        elif not gate.allows_motion:
            desired = (0.0, 0.0, 0.0)
            reason = "state_gate"
        elif not raw_command.valid:
            desired = (0.0, 0.0, 0.0)
        elif self.joystick_inhibited:
            desired = (0.0, 0.0, 0.0)
            reason = "joystick_inhibited"

        limited = self._rate_limit(desired, stamp)
        return BaseCommand(
            limited[0],
            limited[1],
            limited[2],
            stamp=stamp,
            source=raw_command.source,
            valid=valid and gate.allows_motion and not self.dry_run,
            inhibited=self.joystick_inhibited or self.dry_run or not gate.allows_motion,
            reason=reason,
        )

    def _rate_limit(self, desired: Tuple[float, float, float], now: float) -> Tuple[float, float, float]:
        if self.last_update_time < 0.0:
            self.last_command = tuple(float(value) for value in desired)
            self.last_update_time = now
            return self.last_command
        dt = max(now - self.last_update_time, 0.0)
        self.last_update_time = now
        limited = []
        for previous, target, limit in zip(self.last_command, desired, self.acc_limits):
            max_delta = limit * dt
            delta = max(-max_delta, min(max_delta, target - previous))
            limited.append(previous + delta)
        self.last_command = (limited[0], limited[1], limited[2])
        return self.last_command


def _validate_axis(axis: str, name: str) -> str:
    if axis not in JOYSTICK_AXES:
        raise ValueError(f"{name} must be one of {JOYSTICK_AXES}, got {axis!r}")
    return axis


def _validate_sign(sign: int, name: str) -> int:
    sign = int(sign)
    if sign not in (-1, 1):
        raise ValueError(f"{name} must be -1 or 1, got {sign!r}")
    return sign


def _finite_float(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _nonnegative_float(value: float, name: str) -> float:
    value = _finite_float(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return value


def _positive_float(value: float, name: str) -> float:
    value = _finite_float(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return value


def _clip_axis(value: float) -> float:
    value = _finite_float(value, "axis")
    return max(-1.0, min(1.0, value))
