from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import socket
import time
from typing import Iterable, Optional, Tuple
import uuid

from modules.base_command_provider import (
    BaseCommand,
    BaseCommandGate,
    CommandSafetyFilter,
    WirelessJoystickCommandProvider,
)
from modules.safety_lease import SafetyHeartbeat


SPORT_API_ID_STOP_MOVE = 1003
SPORT_API_ID_MOVE = 1008
SPORT_API_ID_SWITCH_JOYSTICK = 1027
OBSTACLE_AVOID_API_ID_SWITCH_SET = 1001
OBSTACLE_AVOID_API_ID_SWITCH_GET = 1002

SPORT_REQUEST_TOPIC = "/api/sport/request"
OBSTACLE_AVOID_REQUEST_TOPIC = "/api/obstacles_avoid/request"
OBSTACLE_AVOID_RESPONSE_TOPIC = "/api/obstacles_avoid/response"
WIRELESS_CONTROLLER_TOPIC = "/wirelesscontroller"
LOW_COMMAND_TOPIC = "/lowcmd"
SPORT_HARD_LIMITS = (0.3, 0.2, 0.3)


def _finite_triplet(values: Iterable[float], name: str) -> Tuple[float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain three finite values")
    return result


def validate_command_limits(limits: Iterable[float]) -> Tuple[float, float, float]:
    result = _finite_triplet(limits, "sport command limits")
    if any(limit < 0.0 for limit in result):
        raise ValueError("sport command limits must be nonnegative")
    for name, limit, hard_limit in zip(
        ("vx", "vy", "yaw_rate"), result, SPORT_HARD_LIMITS
    ):
        if limit > hard_limit:
            raise ValueError(
                f"sport command limit {name}={limit:.6f} exceeds hard limit "
                f"{hard_limit:.6f}"
            )
    return result


def sport_move_parameter(command: Iterable[float]) -> str:
    vx, vy, yaw_rate = _finite_triplet(command, "sport Move command")
    return json.dumps(
        {"x": vx, "y": vy, "z": yaw_rate},
        sort_keys=True,
        separators=(",", ":"),
    )


def boolean_parameter(value: bool, *, field: str) -> str:
    return json.dumps({field: bool(value)}, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class JoystickConfig:
    vx_axis: str = "ly"
    vx_sign: int = 1
    vy_axis: str = "lx"
    vy_sign: int = -1
    yaw_axis: str = "rx"
    yaw_sign: int = -1
    deadzone: float = 0.12
    max_vx: float = 0.30
    max_vy: float = 0.0
    max_yaw: float = 0.30
    watchdog_sec: float = 0.25
    acc_vx: float = 0.30
    acc_vy: float = 0.30
    acc_yaw: float = 0.60


class SportModeCommandSource:
    """Maps only wireless-controller axes to bounded SportMode commands."""

    def __init__(self, config: JoystickConfig):
        validate_command_limits((config.max_vx, config.max_vy, config.max_yaw))
        self.provider = WirelessJoystickCommandProvider(
            vx_axis=config.vx_axis,
            vx_sign=config.vx_sign,
            vy_axis=config.vy_axis,
            vy_sign=config.vy_sign,
            yaw_axis=config.yaw_axis,
            yaw_sign=config.yaw_sign,
            deadzone=config.deadzone,
            max_vx=config.max_vx,
            max_vy=config.max_vy,
            max_yaw=config.max_yaw,
            watchdog_sec=config.watchdog_sec,
        )
        self.filter = CommandSafetyFilter(
            acc_vx=config.acc_vx,
            acc_vy=config.acc_vy,
            acc_yaw=config.acc_yaw,
        )
        self.center_required = True

    def observe_axes(
        self,
        *,
        lx: float,
        ly: float,
        rx: float,
        ry: float,
        stamp: Optional[float] = None,
    ) -> None:
        self.provider.update_wireless(
            lx=lx,
            ly=ly,
            rx=rx,
            ry=ry,
            stamp=stamp,
        )

    def update(self, *, now: Optional[float] = None) -> BaseCommand:
        stamp = time.monotonic() if now is None else float(now)
        raw = self.provider.update(now=stamp)
        if not raw.valid:
            self.center_required = True
            self.filter.reset(now=stamp)
            return raw

        if self.center_required:
            self.filter.reset(now=stamp)
            if self.provider.axes_centered():
                self.center_required = False
            return BaseCommand(
                0.0,
                0.0,
                0.0,
                stamp=stamp,
                source="wireless_joystick",
                valid=False,
                inhibited=True,
                reason="joystick_center_required",
            )

        gate = BaseCommandGate(
            standup_done=True,
            policy_running=True,
            lowlevel_align_done=True,
        )
        return self.filter.update(
            raw,
            gate,
            axes_centered=self.provider.axes_centered(),
            now=stamp,
        )


class SportModeWirelessNode:
    def __init__(
        self,
        *,
        joystick_config: JoystickConfig,
        control_hz: float = 20.0,
        startup_timeout_sec: float = 15.0,
        safety_topic: str = "/safety/estop",
        safety_heartbeat_topic: str = "/safety/heartbeat",
    ) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from std_msgs.msg import Bool, String
        from unitree_api.msg import Request, Response
        from unitree_go.msg import WirelessController

        control_hz = float(control_hz)
        startup_timeout_sec = float(startup_timeout_sec)
        if not math.isfinite(control_hz) or control_hz <= 0.0:
            raise ValueError("control_hz must be positive")
        if not math.isfinite(startup_timeout_sec) or startup_timeout_sec <= 0.0:
            raise ValueError("startup_timeout_sec must be positive")

        self.rclpy = rclpy
        self.Request = Request
        self.Bool = Bool
        self.String = String
        self.node = Node("sportmode_wireless_node")
        self.command_source = SportModeCommandSource(joystick_config)
        self.control_hz = control_hz
        self.startup_timeout_sec = startup_timeout_sec
        self.safety_topic = str(safety_topic)
        self.safety_heartbeat_topic = str(safety_heartbeat_topic)
        self.started_at = time.monotonic()
        self.session_id = str(uuid.uuid4())
        self.hostname = socket.gethostname()
        self.request_id = 0
        self.heartbeat_sequence = 0
        self.obstacle_phase = "set"
        self.obstacle_avoidance_disabled = False
        self.last_obstacle_request_time = -1.0
        self.joystick_disable_count = 0
        self.ready = False
        self.fatal_error: Optional[str] = None
        self.should_exit = False
        self.exit_after = -1.0
        self._shutdown_complete = False
        self.last_command_reason = "startup"

        safety_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.sport_request_pub = self.node.create_publisher(
            Request, SPORT_REQUEST_TOPIC, 10
        )
        self.obstacle_request_pub = self.node.create_publisher(
            Request, OBSTACLE_AVOID_REQUEST_TOPIC, 10
        )
        self.estop_pub = self.node.create_publisher(Bool, self.safety_topic, safety_qos)
        self.heartbeat_pub = self.node.create_publisher(
            String, self.safety_heartbeat_topic, safety_qos
        )
        self.wireless_sub = self.node.create_subscription(
            WirelessController,
            WIRELESS_CONTROLLER_TOPIC,
            self._wireless_callback,
            1,
        )
        self.obstacle_response_sub = self.node.create_subscription(
            Response,
            OBSTACLE_AVOID_RESPONSE_TOPIC,
            self._obstacle_response_callback,
            10,
        )
        self.control_timer = self.node.create_timer(
            1.0 / self.control_hz, self._control_timer_callback
        )
        self.heartbeat_timer = self.node.create_timer(0.1, self._publish_heartbeat)

        self.node.get_logger().info(
            "Pure SportMode: wireless axes -> Move(vx, vy, yaw); no policy or lowcmd"
        )
        self.node.get_logger().info(
            "Wireless buttons are ignored; lateral velocity limit is "
            f"{joystick_config.max_vy:.3f}"
        )
        self.node.get_logger().info(
            "Waiting to confirm obstacles_avoid=false and disable factory joystick handling"
        )

    def _next_request(self, api_id: int, parameter: str = "", *, noreply: bool):
        self.request_id += 1
        request = self.Request()
        request.header.identity.id = self.request_id
        request.header.identity.api_id = int(api_id)
        request.header.lease.id = self.request_id
        request.header.policy.priority = 0
        request.header.policy.noreply = bool(noreply)
        request.parameter = str(parameter)
        return request

    def _wireless_callback(self, msg) -> None:
        try:
            self.command_source.observe_axes(
                lx=float(msg.lx),
                ly=float(msg.ly),
                rx=float(msg.rx),
                ry=float(msg.ry),
            )
        except (TypeError, ValueError) as exc:
            self._trigger_fatal(f"invalid wireless-controller axes: {exc}")

    def _obstacle_response_callback(self, msg) -> None:
        api_id = int(msg.header.identity.api_id)
        if api_id not in {
            OBSTACLE_AVOID_API_ID_SWITCH_SET,
            OBSTACLE_AVOID_API_ID_SWITCH_GET,
        }:
            return
        status_code = int(msg.header.status.code)
        if status_code != 0:
            self._trigger_fatal(
                f"obstacles_avoid API {api_id} returned status {status_code}"
            )
            return
        if api_id == OBSTACLE_AVOID_API_ID_SWITCH_SET:
            self.obstacle_phase = "get"
            self.last_obstacle_request_time = -1.0
            return
        try:
            payload = json.loads(str(msg.data))
            enabled = payload["enable"]
            if not isinstance(enabled, bool):
                raise TypeError("enable must be a boolean")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._trigger_fatal(f"invalid obstacles_avoid status response: {exc}")
            return
        if enabled:
            self._trigger_fatal("obstacle avoidance remained enabled after disable request")
            return
        self.obstacle_avoidance_disabled = True
        self.obstacle_phase = "done"
        self.node.get_logger().info("Confirmed Unitree obstacle avoidance is disabled")

    def _send_obstacle_request_if_due(self, now: float) -> None:
        if self.obstacle_avoidance_disabled:
            return
        if self.node.count_subscribers(OBSTACLE_AVOID_REQUEST_TOPIC) < 1:
            return
        if self.last_obstacle_request_time >= 0.0 and now - self.last_obstacle_request_time < 1.0:
            return
        if self.obstacle_phase == "set":
            request = self._next_request(
                OBSTACLE_AVOID_API_ID_SWITCH_SET,
                boolean_parameter(False, field="enable"),
                noreply=False,
            )
        else:
            request = self._next_request(
                OBSTACLE_AVOID_API_ID_SWITCH_GET,
                noreply=False,
            )
        self.obstacle_request_pub.publish(request)
        self.last_obstacle_request_time = now

    def _send_sport_request(
        self,
        api_id: int,
        parameter: str = "",
        *,
        noreply: bool = True,
    ) -> None:
        self.sport_request_pub.publish(
            self._next_request(api_id, parameter, noreply=noreply)
        )

    def _send_stop(self) -> None:
        if self.node.count_subscribers(SPORT_REQUEST_TOPIC) > 0:
            self._send_sport_request(SPORT_API_ID_STOP_MOVE)

    def _advance_preflight(self, now: float) -> None:
        self._send_obstacle_request_if_due(now)
        if now - self.started_at > self.startup_timeout_sec:
            self._trigger_fatal(
                "startup timed out before obstacle avoidance was confirmed off and "
                "the SportMode endpoint became ready"
            )
            return
        if self.node.count_subscribers(SPORT_REQUEST_TOPIC) > 0:
            self._send_stop()
            if self.joystick_disable_count < 3:
                self._send_sport_request(
                    SPORT_API_ID_SWITCH_JOYSTICK,
                    boolean_parameter(False, field="data"),
                )
                self.joystick_disable_count += 1
        if not self.obstacle_avoidance_disabled or self.joystick_disable_count < 3:
            return
        self.ready = True
        self.node.get_logger().info(
            "Pure SportMode ready; center the sticks once before motion is accepted"
        )

    def _control_timer_callback(self) -> None:
        now = time.monotonic()
        if self.fatal_error is None and self.node.count_publishers(LOW_COMMAND_TOPIC) > 0:
            self._trigger_fatal(
                "detected a lowcmd publisher; pure SportMode cannot run with low-level control"
            )
        if self.fatal_error is not None:
            self._send_stop()
            if now >= self.exit_after:
                self.should_exit = True
            return
        if not self.ready:
            self._advance_preflight(now)
            return

        command = self.command_source.update(now=now)
        if command.valid:
            self._send_sport_request(
                SPORT_API_ID_MOVE,
                sport_move_parameter(command.as_tuple()),
            )
        else:
            self._send_stop()
        if command.reason != self.last_command_reason:
            self.node.get_logger().info(
                f"SportMode command state: {command.reason or 'active'}"
            )
            self.last_command_reason = command.reason

    def _publish_heartbeat(self) -> None:
        self.heartbeat_sequence += 1
        if self.fatal_error is not None:
            state = "FAULT"
        elif self.ready:
            state = "SPORTMODE_ACTIVE"
        else:
            state = "PREFLIGHT"
        heartbeat = SafetyHeartbeat(
            source_pid=os.getpid(),
            source_host=self.hostname,
            session_id=self.session_id,
            sequence=self.heartbeat_sequence,
            sent_monotonic=time.monotonic(),
            safety_state=state,
            estop_latched=self.fatal_error is not None,
        )
        message = self.String()
        message.data = heartbeat.to_json()
        self.heartbeat_pub.publish(message)

    def _trigger_fatal(self, reason: str) -> None:
        if self.fatal_error is not None:
            return
        self.fatal_error = str(reason)
        self.ready = False
        self.exit_after = time.monotonic() + 0.25
        self.node.get_logger().error(self.fatal_error)
        self._send_stop()
        message = self.Bool()
        message.data = True
        self.estop_pub.publish(message)
        self._publish_heartbeat()

    def shutdown(self) -> None:
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        for _ in range(5):
            self._send_stop()
            time.sleep(0.02)
        self.node.destroy_node()
