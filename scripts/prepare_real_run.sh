#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GX_REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/prepare_flat_run.sh [options]
  scripts/prepare_rough_run.sh [options]

This wraps the repeated pre-run work before starting the real control nodes:
  - build Unitree ROS2 messages, robot_state messages, and the sport-mode tool
  - source gx-real runtime environment and run scripts/check_env.sh
  - optionally check SpaceMouse diagnostics when explicitly requested
  - ensure can0 is ready for X5/ARX5
  - check Go2 ROS2 topics and verify/release MCF
  - reject startup if another WBC/X5 writer is already running

Options:
  --network-iface IFACE     Go2 network interface for sport-mode disable. Default: eth0
  --can-dev DEV             USB-CAN serial device, or auto. Default: auto
  --can-if IFACE            SocketCAN interface. Default: can0
  --slcan-speed-code CODE   slcand speed code. Default: 8
  --topic-timeout SEC       Timeout for each ROS2 topic probe. Default: 4
  --no-build                Skip colcon/cmake builds.
  --no-can                  Skip CAN setup/check.
  --force-can-setup         Re-run setup_arx_can.sh even if the CAN interface is UP.
  --no-disable-sport-mode   Skip disable_sports_mode_go2.sh.
  --spacemouse              Check legacy SpaceMouse diagnostics (not used by production fixed-hold).
  --no-spacemouse           Skip SpaceMouse-specific checks. Default.
  --skip-go2-topics         Skip Go2 ROS2 topic checks.
  --check-joystick-motion   Ask the operator to move sticks and verify lx/ly/rx/ry change.
  --joystick-motion-timeout SEC
                            Sampling window for --check-joystick-motion. Default: 6
  --joystick-motion-threshold VALUE
                            Axis magnitude required for --check-joystick-motion. Default: 0.20
  --allow-non-jetson        Do not fail when uname -m is not aarch64.
  -h, --help                Show this help.

This file is the shared implementation. Direct invocation is blocked because
the policy kind would be ambiguous.

After a mode-specific script succeeds, start the external X5 owner and exactly
one matching leg deployment entrypoint.
  scripts/run_x5_fixed_hold_flat.sh + scripts/run_leg12_flat_real.sh
  scripts/run_x5_fixed_hold_rough.sh + scripts/run_leg12_rough_real.sh
EOF
}

BUILD=1
CHECK_CAN=1
FORCE_CAN_SETUP=0
DISABLE_SPORT_MODE=1
CHECK_SPACEMOUSE=0
CHECK_GO2_TOPICS=1
CHECK_JOYSTICK_MOTION=0
REQUIRE_JETSON=1
DEPLOYMENT_KIND="${GX_REAL_DEPLOYMENT_KIND:-}"

NETWORK_IFACE="${GX_REAL_NETWORK_IFACE:-eth0}"
CAN_DEV="${CAN_DEV:-auto}"
CAN_IF="${CAN_IF:-can0}"
SLCAN_SPEED_CODE="${SLCAN_SPEED_CODE:-8}"
ROS_TOPIC_TIMEOUT="${ROS_TOPIC_TIMEOUT:-4}"
JOYSTICK_MOTION_TIMEOUT="${JOYSTICK_MOTION_TIMEOUT:-6}"
JOYSTICK_MOTION_THRESHOLD="${JOYSTICK_MOTION_THRESHOLD:-0.20}"
ROUGH_LIDAR_POINTS_TOPIC="${GX_REAL_ROUGH_LIDAR_POINTS_TOPIC:-/lidar/points_deskewed}"
ROUGH_LIDAR_IMU_TOPIC="${GX_REAL_ROUGH_LIDAR_IMU_TOPIC:-/lidar/imu_raw}"
ROUGH_HEIGHT_TOPIC="${GX_REAL_ROUGH_HEIGHT_TOPIC:-/terrain/elevation_map}"
ROUGH_POSE_TOPIC="${GX_REAL_ROUGH_POSE_TOPIC:-/localization/pose}"

info() {
  printf '[gx-real] %s\n' "$*"
}

warn() {
  printf '[gx-real] warning: %s\n' "$*" >&2
}

die() {
  printf '[gx-real] ERROR: %s\n' "$*" >&2
  exit 1
}

need_arg() {
  local opt="$1"
  local value="${2:-}"
  [[ -n "${value}" ]] || die "${opt} requires an argument"
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --network-iface)
      need_arg "$1" "${2:-}"
      NETWORK_IFACE="$2"
      shift 2
      ;;
    --can-dev)
      need_arg "$1" "${2:-}"
      CAN_DEV="$2"
      shift 2
      ;;
    --can-if)
      need_arg "$1" "${2:-}"
      CAN_IF="$2"
      shift 2
      ;;
    --slcan-speed-code)
      need_arg "$1" "${2:-}"
      SLCAN_SPEED_CODE="$2"
      shift 2
      ;;
    --topic-timeout)
      need_arg "$1" "${2:-}"
      ROS_TOPIC_TIMEOUT="$2"
      shift 2
      ;;
    --no-build)
      BUILD=0
      shift
      ;;
    --no-can)
      CHECK_CAN=0
      shift
      ;;
    --force-can-setup)
      FORCE_CAN_SETUP=1
      shift
      ;;
    --no-disable-sport-mode)
      DISABLE_SPORT_MODE=0
      shift
      ;;
    --spacemouse)
      CHECK_SPACEMOUSE=1
      shift
      ;;
    --no-spacemouse)
      CHECK_SPACEMOUSE=0
      shift
      ;;
    --skip-go2-topics)
      CHECK_GO2_TOPICS=0
      shift
      ;;
    --check-joystick-motion)
      CHECK_JOYSTICK_MOTION=1
      shift
      ;;
    --joystick-motion-timeout)
      need_arg "$1" "${2:-}"
      JOYSTICK_MOTION_TIMEOUT="$2"
      shift 2
      ;;
    --joystick-motion-threshold)
      need_arg "$1" "${2:-}"
      JOYSTICK_MOTION_THRESHOLD="$2"
      shift 2
      ;;
    --allow-non-jetson)
      REQUIRE_JETSON=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

require_command() {
  local command_name="$1"
  command -v "${command_name}" >/dev/null 2>&1 || die "required command not found: ${command_name}"
}

timeout_arg() {
  if [[ "${ROS_TOPIC_TIMEOUT}" =~ [a-zA-Z]$ ]]; then
    printf '%s\n' "${ROS_TOPIC_TIMEOUT}"
  else
    printf '%ss\n' "${ROS_TOPIC_TIMEOUT}"
  fi
}

source_ros_setup() {
  if [[ -f /opt/ros/foxy/setup.bash ]]; then
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/foxy/setup.bash
    set -u
    info "sourced ROS2: /opt/ros/foxy/setup.bash"
  elif [[ -f /opt/ros/humble/setup.bash ]]; then
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    set -u
    info "sourced ROS2: /opt/ros/humble/setup.bash"
  else
    die "missing ROS2 setup.bash under /opt/ros/foxy or /opt/ros/humble"
  fi
}

source_gx_env() {
  # shellcheck disable=SC1091
  source "${GX_REAL_ROOT}/scripts/setup_env.sh"
}

check_host() {
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    die "conda is active (${CONDA_PREFIX}); run 'conda deactivate' before real deployment"
  fi

  local arch
  arch="$(uname -m)"
  if [[ "${arch}" != "aarch64" ]]; then
    if [[ "${REQUIRE_JETSON}" -eq 1 ]]; then
      die "expected Jetson/aarch64 for real deployment, got ${arch}; pass --allow-non-jetson only for offline diagnostics"
    fi
    warn "non-Jetson architecture accepted for diagnostics: ${arch}"
  fi
}

build_unitree_ros2() {
  require_command colcon
  local ws="${GX_REAL_ROOT}/unitree_ros2/cyclonedds_ws"
  [[ -d "${ws}" ]] || die "missing Unitree ROS2 workspace: ${ws}"

  info "building Unitree ROS2 message packages"
  pushd "${ws}" >/dev/null
  colcon build \
    --cmake-clean-cache \
    --packages-select unitree_api unitree_go unitree_hg \
    --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
  popd >/dev/null
}

build_robot_state() {
  require_command colcon
  local ws="${GX_REAL_ROOT}/real-wbc/ros2"
  [[ -d "${ws}/robot_state" ]] || die "missing robot_state package under ${ws}"

  info "building robot_state ROS2 messages"
  pushd "${ws}" >/dev/null
  colcon build \
    --cmake-clean-cache \
    --packages-select robot_state \
    --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
  popd >/dev/null
}

build_sport_mode_tool() {
  require_command cmake
  local sdk_dir="${GX_REAL_ROOT}/unitree_sdk2"
  local build_dir="${GX_REAL_SDK_BUILD_DIR:-${sdk_dir}/build}"
  local cache_path="${build_dir}/CMakeCache.txt"
  if [[ -z "${GX_REAL_SDK_BUILD_DIR:-}" && -f "${cache_path}" ]] \
    && ! grep -Fqx "CMAKE_HOME_DIRECTORY:INTERNAL=${sdk_dir}" "${cache_path}"; then
    warn "unitree_sdk2/build belongs to another source path; preserving it and using build-gx-real"
    build_dir="${sdk_dir}/build-gx-real"
  fi
  [[ -f "${sdk_dir}/CMakeLists.txt" ]] || die "missing unitree_sdk2 CMakeLists.txt"

  local jobs
  jobs="$(nproc 2>/dev/null || printf '4')"
  info "building Unitree sport-mode tool"
  cmake -S "${sdk_dir}" -B "${build_dir}"
  cmake --build "${build_dir}" --target disable_sports_mode_go2 -j "${jobs}"
}

run_builds() {
  if [[ "${BUILD}" -eq 0 ]]; then
    info "skipping builds (--no-build)"
    return
  fi

  source_ros_setup
  build_unitree_ros2
  build_robot_state
  build_sport_mode_tool
}

run_environment_check() {
  local args=()
  if [[ "${CHECK_SPACEMOUSE}" -eq 1 ]]; then
    args+=(--spacemouse)
  fi
  if [[ "${DEPLOYMENT_KIND}" == "rough" ]]; then
    args+=(--rough)
  fi

  info "running deployment import/type-support checks"
  "${GX_REAL_ROOT}/scripts/check_env.sh" "${args[@]}"
}

check_ros_middleware() {
  if [[ "${RMW_IMPLEMENTATION:-}" != "rmw_cyclonedds_cpp" ]]; then
    if [[ "${REQUIRE_JETSON}" -eq 0 ]]; then
      warn "ROS2 middleware is ${RMW_IMPLEMENTATION:-unset}; skipping CycloneDDS requirement for non-Jetson diagnostics"
      return
    fi
    die "ROS2 middleware is ${RMW_IMPLEMENTATION:-unset}; gx-real real runs require rmw_cyclonedds_cpp"
  fi
  if [[ -z "${CYCLONEDDS_URI:-}" ]]; then
    if [[ "${REQUIRE_JETSON}" -eq 0 ]]; then
      warn "CYCLONEDDS_URI is unset; skipping CycloneDDS interface check for non-Jetson diagnostics"
      return
    fi
    die "CYCLONEDDS_URI is unset; export GX_REAL_NETWORK_IFACE=${NETWORK_IFACE} before sourcing scripts/setup_env.sh"
  fi
  info "ROS2 middleware is ${RMW_IMPLEMENTATION} on ${NETWORK_IFACE}"
}

check_spacemouse_daemon() {
  if [[ "${CHECK_SPACEMOUSE}" -eq 0 ]]; then
    info "skipping SpaceMouse checks (--no-spacemouse)"
    return
  fi

  require_command pgrep
  if pgrep -x spacenavd >/dev/null 2>&1; then
    info "spacenavd process is running"
    return
  fi

  if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet spacenavd.service; then
    info "spacenavd.service is active"
    return
  fi

  die "spacenavd is not running; start it with 'sudo systemctl start spacenavd.service'"
}

check_spacemouse_input_device() {
  if [[ "${CHECK_SPACEMOUSE}" -eq 0 ]]; then
    return
  fi

  local pattern
  pattern='3Dconnexion|SpaceMouse|Universal Receiver'

  if [[ ! -r /proc/bus/input/devices ]]; then
    die "/proc/bus/input/devices is unavailable; cannot verify the SpaceMouse receiver"
  fi

  if grep -Eiq "${pattern}" /proc/bus/input/devices; then
    info "SpaceMouse input receiver is visible in /proc/bus/input/devices"
    return
  fi

  if [[ -d /dev/input/by-id ]]; then
    info "available /dev/input/by-id entries:"
    ls -l /dev/input/by-id >&2 || true
  else
    warn "/dev/input/by-id does not exist"
  fi

  die "3Dconnexion/SpaceMouse USB receiver not found; plug it into the Jetson and restart spacenavd"
}

check_no_conflicting_control_processes() {
  require_command pgrep
  local pattern
  pattern='[s]pacemouse_teleop.py|[r]un_arm_spacemouse_test.sh|[r]un_spacemouse_arm.py|[r]un_spacemouse_arm.sh|[r]un_x5_fixed_hold(_flat|_rough)?.py|[r]un_x5_fixed_hold_(flat|rough).sh|[r]un_wbc_(flat|rough|leg12).py|[r]un_leg12_(flat|rough)_real.sh|[r]un_wbc.py'

  local matches
  matches="$(pgrep -af "${pattern}" || true)"
  if [[ -n "${matches}" ]]; then
    printf '%s\n' "${matches}" >&2
    die "an existing WBC/X5 writer process is running; stop it before preparing a real run"
  fi
  info "no existing WBC/X5 writer process detected"
}

can_is_up() {
  ip link show "${CAN_IF}" 2>/dev/null | grep -q '<[^>]*UP'
}

print_can_device_candidates() {
  local path
  local found=0
  shopt -s nullglob
  for path in /dev/serial/by-id/* /dev/ttyACM* /dev/ttyUSB*; do
    [[ -e "${path}" ]] || continue
    found=1
    if [[ -L "${path}" ]]; then
      warn "CAN device candidate: ${path} -> $(readlink -f "${path}" 2>/dev/null || true)"
    else
      warn "CAN device candidate: ${path}"
    fi
  done
  shopt -u nullglob
  if [[ "${found}" -eq 0 ]]; then
    warn "no /dev/serial/by-id, /dev/ttyACM*, or /dev/ttyUSB* CAN candidates found"
  fi
}

ensure_can_ready() {
  if [[ "${CHECK_CAN}" -eq 0 ]]; then
    info "skipping CAN setup/check (--no-can)"
    return
  fi

  require_command ip
  if [[ "${FORCE_CAN_SETUP}" -eq 1 ]] || ! can_is_up; then
    info "configuring SocketCAN ${CAN_IF}"
    if ! "${GX_REAL_ROOT}/scripts/setup_arx_can.sh" "${CAN_DEV}" "${CAN_IF}" "${SLCAN_SPEED_CODE}"; then
      print_can_device_candidates
      die "failed to configure ${CAN_IF}; pass --can-dev /dev/serial/by-id/<USB-CAN> or --can-dev /dev/ttyACM0 explicitly"
    fi
  else
    info "SocketCAN ${CAN_IF} is already UP"
  fi

  if ! ip -details link show "${CAN_IF}" >/dev/null; then
    print_can_device_candidates
    die "CAN interface is unavailable after setup: ${CAN_IF}"
  fi
}

ROS_TOPICS_CACHE=""

load_ros_topics() {
  require_command ros2
  require_command timeout
  local timeout_value
  timeout_value="$(timeout_arg)"

  info "checking ROS2 topic list"
  if ! ROS_TOPICS_CACHE="$(timeout "${timeout_value}" ros2 topic list)"; then
    die "failed to list ROS2 topics within ${timeout_value}; check ROS2/CycloneDDS and Go2 network"
  fi
}

find_ros_topic() {
  local candidate
  for candidate in "$@"; do
    if printf '%s\n' "${ROS_TOPICS_CACHE}" | grep -Fx -- "${candidate}" >/dev/null; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

sample_ros_topic() {
  local topic="$1"
  local timeout_value
  local sample_file
  timeout_value="$(timeout_arg)"
  sample_file="$(mktemp)"

  info "sampling ROS2 topic: ${topic}"
  timeout "${timeout_value}" ros2 topic echo "${topic}" >"${sample_file}" || true
  if [[ -s "${sample_file}" ]]; then
    rm -f "${sample_file}"
    return 0
  fi
  rm -f "${sample_file}"
  return 1
}

require_ros_topic() {
  local label="$1"
  shift

  local topic
  if ! topic="$(find_ros_topic "$@")"; then
    printf '%s\n' "${ROS_TOPICS_CACHE}" >&2
    die "missing Go2 ROS2 topic for ${label}; expected one of: $*"
  fi

  sample_ros_topic "${topic}" || die "ROS2 topic ${topic} did not produce a sample; check Go2 network and DDS"
}

require_ros_topic_type() {
  local label="$1"
  local topic="$2"
  local expected_type="$3"
  local timeout_value
  local actual_types
  timeout_value="$(timeout_arg)"

  if ! actual_types="$(timeout "${timeout_value}" ros2 topic type "${topic}")"; then
    die "failed to inspect ROS2 type for ${label} topic ${topic}"
  fi
  if ! printf '%s\n' "${actual_types}" | grep -Fx -- "${expected_type}" >/dev/null; then
    die "wrong ROS2 type for ${label} topic ${topic}: expected ${expected_type}, got ${actual_types}"
  fi
}

check_go2_topics() {
  if [[ "${CHECK_GO2_TOPICS}" -eq 0 ]]; then
    info "skipping Go2 ROS2 topic checks (--skip-go2-topics)"
    return
  fi

  load_ros_topics
  require_ros_topic "lowstate" "/lowstate" "lowstate" "/rt/lowstate" "rt/lowstate"
  require_ros_topic "wireless controller" "/wirelesscontroller" "wirelesscontroller"
}

check_rough_perception_topics() {
  if [[ "${DEPLOYMENT_KIND}" != "rough" ]]; then
    info "flat deployment selected; perception topics are intentionally not required"
    return
  fi

  if [[ -z "${ROS_TOPICS_CACHE}" ]]; then
    load_ros_topics
  fi
  require_ros_topic "deskewed LiDAR points" "${ROUGH_LIDAR_POINTS_TOPIC}"
  require_ros_topic "LiDAR IMU" "${ROUGH_LIDAR_IMU_TOPIC}"
  require_ros_topic "rough elevation map" "${ROUGH_HEIGHT_TOPIC}"
  require_ros_topic "rough localization pose" "${ROUGH_POSE_TOPIC}"
  require_ros_topic_type "deskewed LiDAR points" "${ROUGH_LIDAR_POINTS_TOPIC}" "sensor_msgs/msg/PointCloud2"
  require_ros_topic_type "LiDAR IMU" "${ROUGH_LIDAR_IMU_TOPIC}" "sensor_msgs/msg/Imu"
  require_ros_topic_type "rough elevation map" "${ROUGH_HEIGHT_TOPIC}" "grid_map_msgs/msg/GridMap"
  require_ros_topic_type "rough localization pose" "${ROUGH_POSE_TOPIC}" "geometry_msgs/msg/PoseStamped"
}

check_wireless_joystick_motion() {
  if [[ "${CHECK_JOYSTICK_MOTION}" -eq 0 ]]; then
    return
  fi

  require_command mktemp
  require_command awk
  require_command timeout
  require_command ros2

  local sample_file
  local timeout_value
  sample_file="$(mktemp)"
  timeout_value="${JOYSTICK_MOTION_TIMEOUT}"
  if [[ ! "${timeout_value}" =~ [a-zA-Z]$ ]]; then
    timeout_value="${timeout_value}s"
  fi

  info "move the Go2 joystick sticks now; sampling /wirelesscontroller for ${timeout_value}"
  timeout "${timeout_value}" ros2 topic echo /wirelesscontroller >"${sample_file}" || true

  if awk -v threshold="${JOYSTICK_MOTION_THRESHOLD}" '
    /^[[:space:]]*(lx|ly|rx|ry):/ {
      value = $2 + 0.0
      if (value < 0.0) {
        value = -value
      }
      if (value > max_axis) {
        max_axis = value
      }
    }
    END {
      if (max_axis > threshold) {
        exit 0
      }
      exit 1
    }
  ' "${sample_file}"; then
    info "Go2 joystick axes changed on /wirelesscontroller"
    rm -f "${sample_file}"
    return
  fi

  warn "recent /wirelesscontroller samples:"
  tail -80 "${sample_file}" >&2 || true
  rm -f "${sample_file}"
  die "Go2 joystick axes did not change; check the handheld controller link or lower --joystick-motion-threshold"
}

disable_sport_mode() {
  if [[ "${DISABLE_SPORT_MODE}" -eq 0 ]]; then
    info "skipping sport-mode disable (--no-disable-sport-mode)"
    return
  fi

  info "disabling Go2 sport mode on ${NETWORK_IFACE}"
  "${GX_REAL_ROOT}/scripts/disable_sports_mode_go2.sh" "${NETWORK_IFACE}"
}

print_next_steps() {
  local leg_command
  local arm_command
  if [[ "${DEPLOYMENT_KIND}" == "rough" ]]; then
    leg_command="scripts/run_leg12_rough_real.sh"
    arm_command="scripts/run_x5_fixed_hold_rough.sh"
  else
    leg_command="scripts/run_leg12_flat_real.sh"
    arm_command="scripts/run_x5_fixed_hold_flat.sh"
  fi
  cat <<EOF
[gx-real] ${DEPLOYMENT_KIND} pre-run checks completed.
[gx-real] Start the runtime commands in separate terminals and adjust only those arguments:

Terminal A:
  cd ${GX_REAL_ROOT}
  export GX_REAL_NETWORK_IFACE=${NETWORK_IFACE}
  source scripts/setup_env.sh
  ${arm_command} --model X5 --can-interface ${CAN_IF} --safety-topic /safety/estop

Terminal B:
  cd ${GX_REAL_ROOT}
  export GX_REAL_NETWORK_IFACE=${NETWORK_IFACE}
  source scripts/setup_env.sh
  ${leg_command} --device cpu --pose_estimator none --standup-mode internal --base-command-source wireless_joystick --joy-vx-axis ly --joy-vx-sign 1 --joy-vy-axis lx --joy-vy-sign -1 --joy-yaw-axis rx --joy-yaw-sign -1 --joy-deadzone 0.12 --joy-max-vx 0.50 --joy-max-vy 0.0 --joy-max-yaw 0.0 --arm-control-owner external_fixed_hold --arm-observation-mode fixed_initial --arm-state-topic /arm/state --arm-target-topic /arm/target_state --safety-topic /safety/estop --require-arm-fixed-hold-safety --gripper-cmd 0.0 --leg-kp 200 --leg-kd 10

The actor input is fixed at [0, 0.3, 0.5, 0, 0, 0] for the entire run; the
arm topics above are consumed only by the physical fixed-hold safety gate.

Terminal B verifies/releases MCF, then enters Passive after its first valid
LowState. After it publishes a healthy safety heartbeat and the arm node remains
in STANDBY, explicitly enable the physical fixed-hold target from a third terminal:
  ros2 topic pub --once /arm/fixed_hold/enable std_msgs/msg/Bool "{data: true}"
EOF
}

main() {
  if [[ "${DEPLOYMENT_KIND}" != "flat" && "${DEPLOYMENT_KIND}" != "rough" ]]; then
    die "ambiguous deployment kind; use scripts/prepare_flat_run.sh or scripts/prepare_rough_run.sh"
  fi
  info "deployment_kind=${DEPLOYMENT_KIND}"
  info "root=${GX_REAL_ROOT}"
  check_host
  run_builds
  export GX_REAL_NETWORK_IFACE="${NETWORK_IFACE}"
  source_gx_env
  check_ros_middleware
  run_environment_check
  check_spacemouse_daemon
  check_spacemouse_input_device
  check_no_conflicting_control_processes
  ensure_can_ready
  check_go2_topics
  check_rough_perception_topics
  check_wireless_joystick_motion
  disable_sport_mode
  print_next_steps
}

main "$@"
