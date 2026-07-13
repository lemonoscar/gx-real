from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.hardware_ownership import (  # noqa: E402
    HardwareOwnershipLock,
    validate_lock_dir_identity,
    validate_real_lock_dir,
)


def test_second_writer_fails_and_metadata_has_pid_reuse_guards(tmp_path: Path) -> None:
    first = HardwareOwnershipLock(
        "go2-lowcmd", owner="first", lock_dir=str(tmp_path), real_hardware=False
    )
    second = HardwareOwnershipLock(
        "go2-lowcmd", owner="second", lock_dir=str(tmp_path), real_hardware=False
    )
    first.acquire()
    metadata = json.loads((tmp_path / "go2-lowcmd.lock").read_text(encoding="utf-8"))
    assert metadata["boot_id"]
    assert metadata["process_start_time"]
    assert metadata["filesystem_device"] >= 0
    assert metadata["inode"] > 0
    with pytest.raises(RuntimeError, match="already owned"):
        second.acquire()
    first.release()
    second.acquire()
    second.release()


@pytest.mark.parametrize("path", ["/tmp/gx", str(ROOT / "locks"), "/opt/private-lock"])
def test_real_mode_rejects_non_shared_lock_locations(path: str) -> None:
    with pytest.raises(RuntimeError):
        validate_real_lock_dir(path)


def test_dev_inode_mismatch_is_rejected(tmp_path: Path) -> None:
    actual = validate_lock_dir_identity(str(tmp_path), None)
    validate_lock_dir_identity(str(tmp_path), actual)
    with pytest.raises(RuntimeError, match="dev:ino mismatch"):
        validate_lock_dir_identity(str(tmp_path), "0:0")


def test_sigkill_releases_kernel_flock(tmp_path: Path) -> None:
    code = (
        "import sys,time; sys.path.insert(0, sys.argv[1]); "
        "from modules.hardware_ownership import HardwareOwnershipLock; "
        "lock=HardwareOwnershipLock('x5-can',owner='child',lock_dir=sys.argv[2],real_hardware=False); "
        "lock.acquire(); print('LOCKED', flush=True); time.sleep(30)"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(ROOT / "real-wbc"), str(tmp_path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "LOCKED"
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
        replacement = HardwareOwnershipLock(
            "x5-can", owner="replacement", lock_dir=str(tmp_path), real_hardware=False
        )
        replacement.acquire()
        replacement.release()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
