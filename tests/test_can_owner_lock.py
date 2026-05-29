from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.can_owner_lock import CanOwnerLock  # noqa: E402


def test_can_owner_lock_rejects_second_owner(tmp_path):
    first = CanOwnerLock("can0", owner="first-owner", lock_dir=str(tmp_path))
    second = CanOwnerLock("can0", owner="second-owner", lock_dir=str(tmp_path))

    first.acquire()
    try:
        try:
            second.acquire()
        except RuntimeError as exc:
            assert "first-owner" in str(exc)
        else:
            raise AssertionError("second CAN owner lock unexpectedly acquired")
    finally:
        first.release()

    second.acquire()
    second.release()
