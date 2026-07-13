from __future__ import annotations

import errno
import fcntl
import json
import os
from pathlib import Path
import socket
import time
from typing import Iterable, Optional


DEFAULT_LOCK_DIR = os.environ.get("GX_REAL_LOCK_DIR", "/run/lock/gx-real")
ALLOWED_RESOURCES = frozenset({"go2-lowcmd", "x5-can", "x5-gripper"})
GX_REAL_ROOT = Path(__file__).resolve().parents[2]


class HardwareOwnershipLock:
    def __init__(
        self,
        resource: str,
        *,
        owner: str,
        lock_dir: str = DEFAULT_LOCK_DIR,
        real_hardware: bool = True,
        expected_lock_dir_dev_inode: str | None = None,
    ) -> None:
        self.resource = str(resource)
        if self.resource not in ALLOWED_RESOURCES:
            raise ValueError(f"unsupported hardware resource {self.resource!r}")
        self.owner = str(owner)
        self.lock_dir = os.path.abspath(str(lock_dir))
        self.real_hardware = bool(real_hardware)
        self.expected_lock_dir_dev_inode = (
            expected_lock_dir_dev_inode
            if expected_lock_dir_dev_inode is not None
            else os.environ.get("GX_REAL_LOCK_DIR_DEV_INO")
        )
        self.path = os.path.join(self.lock_dir, f"{self.resource}.lock")
        self._fd: Optional[int] = None
        if self.real_hardware:
            validate_real_lock_dir(self.lock_dir)

    def acquire(self) -> None:
        if self._fd is not None:
            return
        os.makedirs(self.lock_dir, mode=0o775, exist_ok=True)
        validate_lock_dir_identity(self.lock_dir, self.expected_lock_dir_dev_inode)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            owner = _read_metadata(fd)
            os.close(fd)
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise RuntimeError(
                    f"hardware resource {self.resource!r} is already owned by "
                    f"{owner or 'another process'}"
                ) from exc
            raise

        stat = os.fstat(fd)
        metadata = {
            "resource": self.resource,
            "owner": self.owner,
            "pid": os.getpid(),
            "uid": os.getuid(),
            "hostname": socket.gethostname(),
            "boot_id": _read_text("/proc/sys/kernel/random/boot_id"),
            "process_start_time": _process_start_time(os.getpid()),
            "lock_path": self.path,
            "filesystem_device": stat.st_dev,
            "inode": stat.st_ino,
            "acquired_wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        os.ftruncate(fd, 0)
        os.write(fd, (json.dumps(metadata, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(fd)
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        fd = self._fd
        self._fd = None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> "HardwareOwnershipLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class HardwareOwnershipSet:
    def __init__(self, locks: Iterable[HardwareOwnershipLock]) -> None:
        self.locks = tuple(locks)

    def acquire(self) -> None:
        acquired: list[HardwareOwnershipLock] = []
        try:
            for lock in self.locks:
                lock.acquire()
                acquired.append(lock)
        except Exception:
            for lock in reversed(acquired):
                lock.release()
            raise

    def release(self) -> None:
        for lock in reversed(self.locks):
            lock.release()


def validate_real_lock_dir(lock_dir: str) -> None:
    resolved = Path(lock_dir).resolve(strict=False)
    forbidden_roots = (Path("/tmp"), Path("/var/tmp"), GX_REAL_ROOT)
    if any(_is_relative_to(resolved, root) for root in forbidden_roots):
        raise RuntimeError(f"real hardware lock directory is not host-shared: {resolved}")
    if not (
        _is_relative_to(resolved, Path("/run/lock"))
        or _is_relative_to(resolved, Path("/var/lock"))
    ):
        raise RuntimeError(
            f"real hardware lock directory must be host-shared under /run/lock or /var/lock: {resolved}"
        )


def validate_lock_dir_identity(lock_dir: str, expected: str | None) -> str:
    stat = os.stat(lock_dir)
    actual = f"{stat.st_dev}:{stat.st_ino}"
    if expected and actual != str(expected):
        raise RuntimeError(
            f"lock directory dev:ino mismatch: expected {expected}, actual {actual}; "
            "refusing split host/container ownership"
        )
    return actual


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _read_metadata(fd: int) -> str:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        return os.read(fd, 8192).decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return "UNKNOWN"


def _process_start_time(pid: int) -> str:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return fields[21]
    except (OSError, IndexError):
        return "UNKNOWN"
