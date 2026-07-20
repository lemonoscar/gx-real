#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GX_REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GRID_MAP_REPOSITORY="https://github.com/ANYbotics/grid_map.git"
GRID_MAP_BRANCH="foxy-devel"
GRID_MAP_COMMIT="0b8e1acead0db4a6ad680d89d89332cdab73f89f"

ROUGH_POLICY_PATH="${GX_REAL_ROUGH_POLICY_PATH:-${GX_REAL_ROOT}/policies/rough/current/policy.onnx}"
PERCEPTION_WS="${GX_REAL_ROUGH_PERCEPTION_WS:-${HOME}/rough_perception_ws}"
DEFAULT_PERCEPTION_SETUP="${PERCEPTION_WS}/install/setup.bash"
PERCEPTION_SETUP="${GX_REAL_ROUGH_PERCEPTION_SETUP:-${DEFAULT_PERCEPTION_SETUP}}"
PERCEPTION_LAUNCHER="${GX_REAL_ROUGH_PERCEPTION_LAUNCHER:-}"
NETWORK_IFACE="${GX_REAL_NETWORK_IFACE:-eth0}"
CAN_IF="${GX_REAL_CAN_IF:-can0}"
TOPIC_TIMEOUT="${GX_REAL_ROUGH_TOPIC_TIMEOUT:-7}"
MONITOR_DURATION="${GX_REAL_ROUGH_MONITOR_DURATION:-30}"
MONITOR_MIN_VALID_FRAMES="${GX_REAL_ROUGH_MONITOR_MIN_VALID_FRAMES:-5}"
RECORD_DURATION="${GX_REAL_ROUGH_RECORD_DURATION:-10}"

usage() {
  cat <<'EOF'
Usage:
  scripts/rough_real_ops.sh COMMAND [args...]

Safe Rough deployment workflow for ROS 2 Foxy/Jetson:
  install-grid-map   Build pinned grid_map messages/core in the perception workspace.
  probe              Save a read-only host, ROS, LiDAR and process report.
  lidar-check        Require live Unitree point cloud and raw IMU.
  lio-check          Require live Unitree raw IMU, deskewed cloud and localization.
  perception-start   Exec the configured Point-LIO/elevation-map launcher.
  perception-check   Require production topics, types, samples and TF.
  record-raw [SCENE] Record finite Unitree LiDAR/LIO data before mapper setup.
  record [SCENE]     Record a finite raw/derived perception rosbag.
  monitor            Require a trailing run of valid 187D GridMap monitor frames.
  bootstrap          Run install-grid-map, probe and lidar-check; no actuator writers.
  validate           Run perception-check and monitor; no actuator writers.
  preflight          Run validate, then the existing Rough actuator preflight.
  arm                Exec the existing Rough X5 fixed-hold entrypoint.
  legs               Exec the existing Rough leg entrypoint.
  runtime-commands   Print the canonical multi-terminal runtime commands.

Configuration environment variables:
  GX_REAL_ROUGH_PERCEPTION_WS
      Dedicated colcon workspace. Default: ~/rough_perception_ws
  GX_REAL_ROUGH_PERCEPTION_SETUP
      Perception setup.bash. Default: $GX_REAL_ROUGH_PERCEPTION_WS/install/setup.bash
  GX_REAL_ROUGH_PERCEPTION_LAUNCHER
      Executable launcher that owns Point-LIO/elevation mapping. It must publish
      the production topics listed below; shell command strings are not accepted.
  GX_REAL_NETWORK_IFACE
      Unitree DDS interface. Default: eth0
  GX_REAL_CAN_IF
      X5 SocketCAN interface. Default: can0
  GX_REAL_ROUGH_TOPIC_TIMEOUT
      Per-topic liveness window in seconds. Default: 7
  GX_REAL_ROUGH_MONITOR_DURATION
      Finite monitor duration in seconds. Default: 30
  GX_REAL_ROUGH_RECORD_DURATION
      Rosbag duration in seconds. Default: 10
  GX_REAL_OPERATOR_CONFIRM_ACTUATORS=YES
      Required only by preflight, arm and legs.

Production perception contract:
  /lidar/points_deskewed   sensor_msgs/msg/PointCloud2
  /lidar/imu_raw           sensor_msgs/msg/Imu
  /terrain/elevation_map   grid_map_msgs/msg/GridMap, layer=elevation
  /localization/pose       geometry_msgs/msg/PoseStamped
  TF: odom -> base_link and base_link -> lidar

The default bootstrap never publishes LowCmd, configures CAN, releases MCF, or
starts an actuator writer. Point-LIO/elevation mapping must remain a separate
supervised process; this script validates its outputs before actuator preflight.

This script intentionally does not install unitreerobotics/point_lio_unilidar:
that repository is ROS 1 Noetic/catkin, while this robot deployment is ROS 2
Foxy. Configure a reviewed Foxy-native perception launcher instead.
EOF
}

info() {
  printf '[gx-real][rough-ops] %s\n' "$*"
}

warn() {
  printf '[gx-real][rough-ops] warning: %s\n' "$*" >&2
}

die() {
  printf '[gx-real][rough-ops] ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  local name="$1"
  command -v "${name}" >/dev/null 2>&1 || die "required command not found: ${name}"
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer, got ${value}"
}

source_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "missing setup file: ${path}"
  set +u
  # shellcheck disable=SC1090
  source "${path}"
  set -u
}

source_foxy() {
  [[ -f /opt/ros/foxy/setup.bash ]] || die "this workflow requires /opt/ros/foxy/setup.bash"
  source_file /opt/ros/foxy/setup.bash
  [[ "${ROS_DISTRO:-}" == "foxy" ]] || die "expected ROS_DISTRO=foxy, got ${ROS_DISTRO:-unset}"
}

configure_perception_setup() {
  if [[ -n "${GX_REAL_ROUGH_PERCEPTION_SETUP:-}" ]]; then
    [[ -f "${PERCEPTION_SETUP}" ]] || die "missing GX_REAL_ROUGH_PERCEPTION_SETUP: ${PERCEPTION_SETUP}"
    export GX_REAL_PERCEPTION_SETUP="${PERCEPTION_SETUP}"
  elif [[ -f "${DEFAULT_PERCEPTION_SETUP}" ]]; then
    export GX_REAL_PERCEPTION_SETUP="${DEFAULT_PERCEPTION_SETUP}"
  else
    unset GX_REAL_PERCEPTION_SETUP || true
  fi
}

source_rough_environment() {
  export GX_REAL_POLICY_PATH="${ROUGH_POLICY_PATH}"
  export GX_REAL_NETWORK_IFACE="${NETWORK_IFACE}"
  configure_perception_setup
  # shellcheck disable=SC1091
  source "${GX_REAL_ROOT}/scripts/setup_env.sh"
}

require_perception_environment() {
  [[ -f "${PERCEPTION_SETUP}" ]] || die \
    "missing perception setup ${PERCEPTION_SETUP}; run install-grid-map or set GX_REAL_ROUGH_PERCEPTION_SETUP"
  export GX_REAL_ROUGH_PERCEPTION_SETUP="${PERCEPTION_SETUP}"
  source_rough_environment
  "${GX_REAL_PYTHON_BIN}" -c \
    'from grid_map_msgs.msg import GridMap; print("[gx-real][rough-ops] GridMap import ready")'
}

check_no_control_writers() {
  require_command pgrep
  local pattern
  pattern='[r]un_x5_fixed_hold(_flat|_rough)?.py|[r]un_wbc_(flat|rough|leg12).py|[r]un_leg12_(flat|rough)_real.sh|[r]un_wbc.py'
  local matches
  matches="$(pgrep -af "${pattern}" || true)"
  if [[ -n "${matches}" ]]; then
    printf '%s\n' "${matches}" >&2
    die "an actuator writer is already running"
  fi
}

install_grid_map() {
  require_command git
  require_command colcon
  source_foxy

  [[ -f /usr/include/eigen3/Eigen/Core ]] || die \
    "missing Eigen headers; install libeigen3-dev, then rerun install-grid-map"

  local source_dir="${PERCEPTION_WS}/src/grid_map"
  mkdir -p "${PERCEPTION_WS}/src"

  if [[ -e "${source_dir}" && ! -d "${source_dir}/.git" ]]; then
    die "existing path is not a Git checkout: ${source_dir}"
  fi

  if [[ ! -d "${source_dir}/.git" ]]; then
    info "cloning grid_map ${GRID_MAP_BRANCH} into ${source_dir}"
    git clone --branch "${GRID_MAP_BRANCH}" "${GRID_MAP_REPOSITORY}" "${source_dir}"
  else
    local remote_url
    remote_url="$(git -C "${source_dir}" remote get-url origin 2>/dev/null || true)"
    case "${remote_url}" in
      https://github.com/ANYbotics/grid_map.git|git@github.com:ANYbotics/grid_map.git)
        ;;
      *)
        die "unexpected grid_map origin at ${source_dir}: ${remote_url:-unset}"
        ;;
    esac
    if [[ -n "$(git -C "${source_dir}" status --porcelain)" ]]; then
      die "grid_map checkout has local changes; preserve or remove them before continuing"
    fi
    info "refreshing pinned grid_map source"
    git -C "${source_dir}" fetch --depth 1 origin "${GRID_MAP_BRANCH}"
  fi

  if ! git -C "${source_dir}" cat-file -e "${GRID_MAP_COMMIT}^{commit}" 2>/dev/null; then
    git -C "${source_dir}" fetch --depth 1 origin "${GRID_MAP_COMMIT}"
  fi
  git -C "${source_dir}" checkout --detach "${GRID_MAP_COMMIT}"
  [[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${GRID_MAP_COMMIT}" ]] || \
    die "grid_map commit verification failed"

  info "building GridMap messages/core at ${GRID_MAP_COMMIT}"
  pushd "${PERCEPTION_WS}" >/dev/null
  colcon build \
    --merge-install \
    --packages-select grid_map_cmake_helpers grid_map_core grid_map_msgs \
    --cmake-args \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF \
      -DPython3_EXECUTABLE=/usr/bin/python3
  popd >/dev/null

  source_file "${DEFAULT_PERCEPTION_SETUP}"
  ros2 pkg prefix grid_map_msgs >/dev/null
  /usr/bin/python3 -c \
    'from grid_map_msgs.msg import GridMap; print("[gx-real][rough-ops] GridMap import verified")'
  info "grid_map setup=${DEFAULT_PERCEPTION_SETUP}"
}

topic_type() {
  local topic="$1"
  timeout "${TOPIC_TIMEOUT}s" ros2 topic type "${topic}" 2>/dev/null || true
}

topic_rate_report() {
  local topic="$1"
  local output
  output="$(mktemp)"
  timeout -s INT "${TOPIC_TIMEOUT}s" ros2 topic hz "${topic}" >"${output}" 2>&1 || true
  cat "${output}"
  rm -f "${output}"
}

topic_is_live() {
  local topic="$1"
  local output
  output="$(mktemp)"
  timeout -s INT "${TOPIC_TIMEOUT}s" ros2 topic hz "${topic}" >"${output}" 2>&1 || true
  if grep -q 'average rate:' "${output}"; then
    rm -f "${output}"
    return 0
  fi
  warn "no live samples on ${topic} during ${TOPIC_TIMEOUT}s"
  cat "${output}" >&2
  rm -f "${output}"
  return 1
}

require_topic() {
  local topic="$1"
  local expected_type="$2"
  local actual_type
  actual_type="$(topic_type "${topic}")"
  [[ "${actual_type}" == "${expected_type}" ]] || die \
    "${topic} type mismatch: expected ${expected_type}, got ${actual_type:-missing}"
  topic_is_live "${topic}" || die "required topic is not live: ${topic}"
  info "topic ready: ${topic} (${expected_type})"
}

probe() {
  require_command timeout
  require_command ros2
  require_command tee
  check_no_control_writers
  source_rough_environment

  local stamp
  local output_dir
  local report
  stamp="$(date +%Y%m%d-%H%M%S)"
  output_dir="${GX_REAL_ROOT}/logs/rough_probe/${stamp}"
  report="${output_dir}/report.txt"
  mkdir -p "${output_dir}"

  {
    printf '===== TIME AND SYSTEM =====\n'
    date --iso-8601=seconds
    uname -a
    cat /etc/os-release
    cat /etc/nv_tegra_release 2>/dev/null || true
    free -h
    df -h "${HOME}"

    printf '===== NETWORK =====\n'
    ip -br link 2>&1 || true
    ip -br addr show "${NETWORK_IFACE}" 2>&1 || true
    ip -details link show "${CAN_IF}" 2>&1 || true

    printf '===== REPOSITORY =====\n'
    git -C "${GX_REAL_ROOT}" status --short --branch
    git -C "${GX_REAL_ROOT}" rev-parse HEAD
    sha256sum "${ROUGH_POLICY_PATH}"
    sha256sum "${GX_REAL_ROOT}/policies/rough/current/model_37500.pt"

    printf '===== ENVIRONMENT =====\n'
    printf 'ROS_DISTRO=%s\n' "${ROS_DISTRO:-unset}"
    printf 'RMW_IMPLEMENTATION=%s\n' "${RMW_IMPLEMENTATION:-unset}"
    printf 'GX_REAL_PERCEPTION_SETUP=%s\n' "${GX_REAL_PERCEPTION_SETUP:-unset}"
    /usr/bin/python3 --version
    /usr/bin/python3 -c \
      'import onnxruntime; print("onnxruntime", onnxruntime.__version__)' 2>&1 || true
    /usr/bin/python3 -c \
      'from grid_map_msgs.msg import GridMap; print("GridMap", GridMap)' 2>&1 || true

    printf '===== ROS GRAPH =====\n'
    timeout 10s ros2 node list 2>&1 || true
    timeout 10s ros2 topic list -t 2>&1 || true
    timeout 20s ros2 doctor --report 2>&1 || true

    printf '===== UNITREE LIDAR INTERFACES =====\n'
    timeout 5s ros2 interface show unitree_go/msg/LidarState 2>&1 || true
    timeout 5s ros2 interface show unitree_go/msg/HeightMap 2>&1 || true

    printf '===== RUNNING PROCESSES =====\n'
    ps -eo pid,user,cmd | \
      grep -Ei 'utlidar|uslam|slam|lio|elevation|grid.map|unitree|wbc|arx' | \
      grep -v grep || true

    for topic in \
      /utlidar/cloud \
      /utlidar/cloud_deskewed \
      /utlidar/cloud_base \
      /utlidar/imu \
      /utlidar/robot_pose \
      /utlidar/robot_odom \
      /utlidar/lidar_state \
      /utlidar/height_map_array \
      /lidar/points_deskewed \
      /lidar/imu_raw \
      /terrain/elevation_map \
      /localization/pose \
      /lowstate \
      /wirelesscontroller
    do
      printf '===== TOPIC %s =====\n' "${topic}"
      timeout 5s ros2 topic info --verbose "${topic}" 2>&1 || true
      topic_rate_report "${topic}"
    done

    printf '===== LIDAR STATE SAMPLE =====\n'
    timeout -s INT 3s ros2 topic echo /utlidar/lidar_state 2>&1 || true

    printf '===== POINT CLOUD HEADER SAMPLE =====\n'
    if ros2 topic echo -h 2>&1 | grep -q -- '--no-arr'; then
      timeout -s INT 3s ros2 topic echo --no-arr /utlidar/cloud 2>&1 || true
      timeout -s INT 3s ros2 topic echo --no-arr /utlidar/cloud_deskewed 2>&1 || true
    else
      printf 'ros2 topic echo has no --no-arr support; use the rosbag instead of dumping cloud data\n'
    fi

    printf '===== POSE SAMPLE =====\n'
    timeout -s INT 3s ros2 topic echo /utlidar/robot_pose 2>&1 || true

    printf '===== TF SAMPLES =====\n'
    timeout -s INT 3s ros2 run tf2_ros tf2_echo base_link lidar 2>&1 || true
    timeout -s INT 3s ros2 run tf2_ros tf2_echo odom base_link 2>&1 || true
  } 2>&1 | tee "${report}"

  info "probe report=${report}"
}

lidar_check() {
  require_command ros2
  require_command timeout
  check_no_control_writers
  source_rough_environment

  require_topic /utlidar/imu sensor_msgs/msg/Imu
  require_topic /utlidar/cloud sensor_msgs/msg/PointCloud2
  info "Unitree LiDAR point cloud and IMU are live"
}

lio_check() {
  lidar_check
  require_topic /utlidar/cloud_deskewed sensor_msgs/msg/PointCloud2
  require_topic /utlidar/robot_pose geometry_msgs/msg/PoseStamped
  info "Unitree onboard deskew/localization outputs are live"
}

perception_start() {
  require_perception_environment
  check_no_control_writers
  [[ -n "${PERCEPTION_LAUNCHER}" ]] || die \
    "set GX_REAL_ROUGH_PERCEPTION_LAUNCHER to an executable Point-LIO/elevation-map launcher"
  [[ -f "${PERCEPTION_LAUNCHER}" ]] || die "missing perception launcher: ${PERCEPTION_LAUNCHER}"
  [[ -x "${PERCEPTION_LAUNCHER}" ]] || die "perception launcher is not executable: ${PERCEPTION_LAUNCHER}"
  info "starting supervised perception launcher: ${PERCEPTION_LAUNCHER}"
  exec "${PERCEPTION_LAUNCHER}" "$@"
}

require_tf() {
  local target="$1"
  local source="$2"
  local output
  output="$(mktemp)"
  timeout -s INT "${TOPIC_TIMEOUT}s" \
    ros2 run tf2_ros tf2_echo "${target}" "${source}" >"${output}" 2>&1 || true
  if grep -Eq 'Translation:|At time ' "${output}"; then
    rm -f "${output}"
    info "TF ready: ${target} <- ${source}"
    return 0
  fi
  cat "${output}" >&2
  rm -f "${output}"
  die "missing TF: ${target} <- ${source}"
}

perception_check() {
  require_command ros2
  require_command timeout
  require_perception_environment
  check_no_control_writers

  require_topic /lidar/points_deskewed sensor_msgs/msg/PointCloud2
  require_topic /lidar/imu_raw sensor_msgs/msg/Imu
  require_topic /terrain/elevation_map grid_map_msgs/msg/GridMap
  require_topic /localization/pose geometry_msgs/msg/PoseStamped
  require_tf base_link lidar
  require_tf odom base_link
  info "production perception topics and TF are ready"
}

record_bag() {
  local mode="$1"
  local scene="${2:-static}"
  [[ "${scene}" =~ ^[A-Za-z0-9._-]+$ ]] || die \
    "SCENE may contain only letters, digits, dot, underscore and dash"
  require_command ros2
  require_command timeout
  require_positive_integer GX_REAL_ROUGH_RECORD_DURATION "${RECORD_DURATION}"

  local topics=(
    /utlidar/cloud
    /utlidar/cloud_deskewed
    /utlidar/cloud_base
    /utlidar/imu
    /utlidar/robot_pose
    /utlidar/robot_odom
    /utlidar/lidar_state
    /utlidar/height_map_array
    /lowstate
  )
  case "${mode}" in
    raw)
      lidar_check
      ;;
    full)
      perception_check
      topics+=(
        /lidar/points_deskewed
        /lidar/imu_raw
        /terrain/elevation_map
        /localization/pose
        /tf
        /tf_static
      )
      ;;
    *)
      die "internal record mode error: ${mode}"
      ;;
  esac

  local stamp
  local output_dir
  local status
  stamp="$(date +%Y%m%d-%H%M%S)"
  output_dir="${GX_REAL_ROOT}/logs/lidar_calibration/${stamp}_${mode}_${scene}"
  mkdir -p "$(dirname "${output_dir}")"

  info "recording ${RECORD_DURATION}s ${mode} perception bag: ${output_dir}"
  set +e
  timeout -s INT "${RECORD_DURATION}s" ros2 bag record \
    -o "${output_dir}" \
    "${topics[@]}"
  status=$?
  set -e
  if [[ "${status}" -ne 0 && "${status}" -ne 124 && "${status}" -ne 130 ]]; then
    die "ros2 bag record failed with status ${status}"
  fi
  [[ -f "${output_dir}/metadata.yaml" ]] || die "rosbag metadata was not finalized: ${output_dir}"
  ros2 bag info "${output_dir}"
  info "rosbag=${output_dir}"
}

monitor() {
  require_command awk
  require_command tee
  require_command timeout
  require_positive_integer GX_REAL_ROUGH_MONITOR_DURATION "${MONITOR_DURATION}"
  require_positive_integer GX_REAL_ROUGH_MONITOR_MIN_VALID_FRAMES "${MONITOR_MIN_VALID_FRAMES}"
  require_perception_environment
  check_no_control_writers

  local stamp
  local output_dir
  local log_path
  local status
  stamp="$(date +%Y%m%d-%H%M%S)"
  output_dir="${GX_REAL_ROOT}/logs/rough_monitor"
  log_path="${output_dir}/${stamp}.log"
  mkdir -p "${output_dir}"

  info "running finite GridMap monitor for ${MONITOR_DURATION}s"
  set +e
  timeout -s INT "${MONITOR_DURATION}s" \
    "${GX_REAL_PYTHON_BIN}" "${GX_REAL_ROOT}/real-wbc/scripts/run_height_scan_monitor.py" \
      --source grid_map \
      --topic /terrain/elevation_map \
      --pose-topic /localization/pose \
      --map-layer elevation \
      --contract "${GX_REAL_ROOT}/policies/rough/current/height_scan_contract.yaml" \
      --timeout 0.25 \
      --min-valid-ratio 0.60 \
      --min-critical-valid-ratio 0.95 \
      --max-critical-sentinel-cells 0 \
      2>&1 | tee "${log_path}"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "${status}" -ne 0 && "${status}" -ne 124 && "${status}" -ne 130 ]]; then
    die "height monitor failed with status ${status}; log=${log_path}"
  fi

  if ! awk -v minimum="${MONITOR_MIN_VALID_FRAMES}" '
    /shape=187/ {
      if ($0 ~ /ok=True/ && $0 ~ /fallback=False/ && $0 ~ /height_source=grid_map/) {
        consecutive += 1
      } else {
        consecutive = 0
      }
    }
    END { exit(consecutive >= minimum ? 0 : 1) }
  ' "${log_path}"; then
    die "monitor did not end with ${MONITOR_MIN_VALID_FRAMES} consecutive valid frames; log=${log_path}"
  fi
  info "Rough read-only monitor passed; log=${log_path}"
}

require_actuator_confirmation() {
  [[ "${GX_REAL_OPERATOR_CONFIRM_ACTUATORS:-}" == "YES" ]] || die \
    "set GX_REAL_OPERATOR_CONFIRM_ACTUATORS=YES only after the robot is safely supported and release evidence is verified"
}

runtime_commands() {
  cat <<EOF
Perception terminal:
  cd ${GX_REAL_ROOT}
  export GX_REAL_ROUGH_PERCEPTION_SETUP=${PERCEPTION_SETUP}
  export GX_REAL_ROUGH_PERCEPTION_LAUNCHER=/absolute/path/to/start_rough_perception.sh
  scripts/rough_real_ops.sh perception-start

Validation terminal:
  cd ${GX_REAL_ROOT}
  export GX_REAL_ROUGH_PERCEPTION_SETUP=${PERCEPTION_SETUP}
  scripts/rough_real_ops.sh validate

Calibration capture terminal, repeat with descriptive scene names:
  cd ${GX_REAL_ROOT}
  export GX_REAL_ROUGH_PERCEPTION_SETUP=${PERCEPTION_SETUP}
  GX_REAL_ROUGH_RECORD_DURATION=30 scripts/rough_real_ops.sh record flat_yaw0

Raw capture before a reviewed mapper is available:
  cd ${GX_REAL_ROOT}
  GX_REAL_ROUGH_RECORD_DURATION=30 scripts/rough_real_ops.sh record-raw prone_inventory

Actuator preflight terminal, only after release contracts are VERIFIED:
  cd ${GX_REAL_ROOT}
  export GX_REAL_ROUGH_PERCEPTION_SETUP=${PERCEPTION_SETUP}
  export GX_REAL_OPERATOR_CONFIRM_ACTUATORS=YES
  scripts/rough_real_ops.sh preflight --network-iface ${NETWORK_IFACE} --can-if ${CAN_IF} --check-joystick-motion

X5 terminal, using the exact command printed by preflight:
  export GX_REAL_ROUGH_PERCEPTION_SETUP=${PERCEPTION_SETUP}
  export GX_REAL_OPERATOR_CONFIRM_ACTUATORS=YES
  scripts/rough_real_ops.sh arm --model X5 --can-interface ${CAN_IF} --safety-topic /safety/estop

Leg terminal, using the exact reviewed arguments printed by preflight:
  export GX_REAL_ROUGH_PERCEPTION_SETUP=${PERCEPTION_SETUP}
  export GX_REAL_OPERATOR_CONFIRM_ACTUATORS=YES
  scripts/rough_real_ops.sh legs [reviewed run_leg12_rough_real.sh arguments]
EOF
}

command="${1:-}"
if [[ -z "${command}" ]]; then
  usage
  exit 2
fi
shift

case "${command}" in
  -h|--help|help)
    usage
    ;;
  install-grid-map)
    [[ "$#" -eq 0 ]] || die "install-grid-map accepts no arguments"
    install_grid_map
    ;;
  probe)
    [[ "$#" -eq 0 ]] || die "probe accepts no arguments"
    probe
    ;;
  lio-check)
    [[ "$#" -eq 0 ]] || die "lio-check accepts no arguments"
    lio_check
    ;;
  lidar-check)
    [[ "$#" -eq 0 ]] || die "lidar-check accepts no arguments"
    lidar_check
    ;;
  perception-start)
    perception_start "$@"
    ;;
  perception-check)
    [[ "$#" -eq 0 ]] || die "perception-check accepts no arguments"
    perception_check
    ;;
  record-raw)
    [[ "$#" -le 1 ]] || die "record-raw accepts at most one SCENE argument"
    record_bag raw "$@"
    ;;
  record)
    [[ "$#" -le 1 ]] || die "record accepts at most one SCENE argument"
    record_bag full "$@"
    ;;
  monitor)
    [[ "$#" -eq 0 ]] || die "monitor accepts no arguments"
    monitor
    ;;
  bootstrap)
    [[ "$#" -eq 0 ]] || die "bootstrap accepts no arguments"
    install_grid_map
    probe
    lidar_check
    ;;
  validate)
    [[ "$#" -eq 0 ]] || die "validate accepts no arguments"
    perception_check
    monitor
    ;;
  preflight)
    require_actuator_confirmation
    perception_check
    monitor
    export GX_REAL_ROUGH_PERCEPTION_SETUP="${PERCEPTION_SETUP}"
    export GX_REAL_PERCEPTION_SETUP="${PERCEPTION_SETUP}"
    exec "${GX_REAL_ROOT}/scripts/prepare_rough_run.sh" "$@"
    ;;
  arm)
    require_actuator_confirmation
    perception_check
    monitor
    export GX_REAL_ROUGH_PERCEPTION_SETUP="${PERCEPTION_SETUP}"
    export GX_REAL_PERCEPTION_SETUP="${PERCEPTION_SETUP}"
    exec "${GX_REAL_ROOT}/scripts/run_x5_fixed_hold_rough.sh" "$@"
    ;;
  legs)
    require_actuator_confirmation
    perception_check
    monitor
    export GX_REAL_ROUGH_PERCEPTION_SETUP="${PERCEPTION_SETUP}"
    export GX_REAL_PERCEPTION_SETUP="${PERCEPTION_SETUP}"
    exec "${GX_REAL_ROOT}/scripts/run_leg12_rough_real.sh" "$@"
    ;;
  runtime-commands)
    [[ "$#" -eq 0 ]] || die "runtime-commands accepts no arguments"
    runtime_commands
    ;;
  *)
    die "unknown command: ${command}; run scripts/rough_real_ops.sh --help"
    ;;
esac
