#!/usr/bin/env bash
set -euo pipefail

COLOR_ERROR='\033[0;31m'
COLOR_SUCCESS='\033[0;32m'
COLOR_WARNING='\033[1;33m'
COLOR_INFO='\033[0;34m'
COLOR_RESET='\033[0m'

print_success() {
    echo -e "${COLOR_SUCCESS}[SUCCESS]${COLOR_RESET} $1"
}

print_warning() {
    echo -e "${COLOR_WARNING}[WARNING]${COLOR_RESET} $1"
}

print_error() {
    echo -e "${COLOR_ERROR}[ERROR]${COLOR_RESET} $1"
}

print_info() {
    echo -e "${COLOR_INFO}[INFO]${COLOR_RESET} $1"
}

print_separator() {
    echo -e "${COLOR_INFO}-------------------------------------------------------------------${COLOR_RESET}"
}

usage() {
    cat <<'EOF'
Usage:
  setup_arx_can.sh [CAN_DEV] [CAN_IF] [SLCAN_SPEED_CODE]

Examples:
  ./scripts/setup_arx_can.sh
  ./scripts/setup_arx_can.sh auto can0 8
  ./scripts/setup_arx_can.sh /dev/ttyACM0 can0 8
  ./scripts/setup_arx_can.sh /dev/serial/by-id/usb-XXXX can0 8

Defaults:
  CAN_DEV=auto
  CAN_IF=can0
  SLCAN_SPEED_CODE=8
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

print_device_candidates() {
    local path
    shopt -s nullglob
    for path in /dev/serial/by-id/* /dev/ttyACM* /dev/ttyUSB*; do
        [[ -e "${path}" ]] || continue
        if [[ -L "${path}" ]]; then
            print_warning "  ${path} -> $(readlink -f "${path}")"
        else
            print_warning "  ${path}"
        fi
    done
    shopt -u nullglob
}

detect_can_dev() {
    local candidates=()
    local path=""
    local resolved=""

    if [[ -d /dev/serial/by-id ]]; then
        shopt -s nullglob
        for path in /dev/serial/by-id/*; do
            [[ -L "${path}" ]] || continue
            resolved="$(readlink -f "${path}" 2>/dev/null || true)"
            case "${resolved}" in
                /dev/ttyACM*|/dev/ttyUSB*)
                    candidates+=("${path}")
                    ;;
            esac
        done
        shopt -u nullglob
        if [[ ${#candidates[@]} -eq 1 ]]; then
            printf '%s\n' "${candidates[0]}"
            return 0
        fi
        if [[ ${#candidates[@]} -gt 1 ]]; then
            print_warning "Multiple /dev/serial/by-id candidates found. Please choose one explicitly:"
            print_device_candidates
            return 1
        fi
    fi

    candidates=()
    shopt -s nullglob
    for path in /dev/ttyACM* /dev/ttyUSB*; do
        [[ -e "${path}" ]] || continue
        candidates+=("${path}")
    done
    shopt -u nullglob

    if [[ ${#candidates[@]} -eq 1 ]]; then
        printf '%s\n' "${candidates[0]}"
        return 0
    fi

    if [[ ${#candidates[@]} -gt 1 ]]; then
        print_warning "Multiple tty candidates found. Please choose one explicitly:"
        print_device_candidates
    else
        print_warning "No ttyACM/ttyUSB device found."
    fi
    return 1
}

CAN_DEV_INPUT="${1:-${CAN_DEV:-auto}}"
CAN_IF="${2:-${CAN_IF:-can0}}"
SLCAN_SPEED_CODE="${3:-${SLCAN_SPEED_CODE:-8}}"
WAIT_RETRIES="${WAIT_RETRIES:-20}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-0.2}"

if [[ -z "${CAN_DEV_INPUT}" || "${CAN_DEV_INPUT}" == "auto" ]]; then
    if ! CAN_DEV="$(detect_can_dev)"; then
        print_error "Unable to auto-detect a unique USB-CAN serial device. Set CAN_DEV explicitly."
        exit 1
    fi
else
    CAN_DEV="${CAN_DEV_INPUT}"
fi

print_separator
print_info "[Setting up ARX SocketCAN]"
print_info "Platform=$(uname -s) arch=$(uname -m)"
print_info "CAN_DEV=${CAN_DEV}"
print_info "CAN_IF=${CAN_IF}"
print_info "SLCAN_SPEED_CODE=${SLCAN_SPEED_CODE}"

if [[ ! -e "${CAN_DEV}" ]]; then
    print_error "Device not found: ${CAN_DEV}"
    exit 1
fi

for cmd in sudo modprobe slcand ip; do
    if ! command -v "${cmd%% *}" >/dev/null 2>&1; then
        print_error "Required command not found: ${cmd%% *}"
        exit 1
    fi
done

print_info "Loading CAN kernel modules..."
sudo modprobe can
sudo modprobe can_raw
sudo modprobe can_dev
sudo modprobe slcan

print_info "Restarting slcand on ${CAN_IF}..."
sudo pkill slcand || true
sudo ip link set "${CAN_IF}" down >/dev/null 2>&1 || true
sudo slcand -o -c -f "-s${SLCAN_SPEED_CODE}" "${CAN_DEV}" "${CAN_IF}"

sleep 1
for _ in $(seq 1 "${WAIT_RETRIES}"); do
    if ip link show "${CAN_IF}" >/dev/null 2>&1; then
        break
    fi
    sleep "${WAIT_INTERVAL_SEC}"
done

if ! ip link show "${CAN_IF}" >/dev/null 2>&1; then
    print_error "Interface ${CAN_IF} was not created by slcand."
    exit 1
fi

print_info "Bringing up ${CAN_IF}..."
sudo ip link set "${CAN_IF}" up

print_success "SocketCAN interface is ready."
print_separator
ip -details link show "${CAN_IF}"
print_separator
ip -s -d link show "${CAN_IF}"
print_separator

if command -v candump >/dev/null 2>&1; then
    print_info "Optional 2s candump probe on ${CAN_IF} (Ctrl+C to skip)..."
    timeout 2s candump "${CAN_IF}" || true
else
    print_warning "candump not found; skip passive bus probe."
fi
