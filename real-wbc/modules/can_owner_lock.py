from __future__ import annotations

import errno
import fcntl
import os
import socket
import time
from typing import Optional


DEFAULT_CAN_LOCK_DIR = os.environ.get("GX_REAL_CAN_LOCK_DIR", "/tmp/gx-real-can-locks")


class CanOwnerLock:
    def __init__(
        self,
        interface: str,
        *,
        owner: str,
        lock_dir: str = DEFAULT_CAN_LOCK_DIR,
    ):
        self.interface = str(interface)
        self.owner = str(owner)
        self.lock_dir = str(lock_dir)
        self.path = os.path.join(
            self.lock_dir,
            f"{_safe_lock_name(self.interface)}.lock",
        )
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        if self._fd is not None:
            return
        os.makedirs(self.lock_dir, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            owner = _read_lock_owner(fd)
            os.close(fd)
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise RuntimeError(
                    f"CAN interface {self.interface!r} is already owned by "
                    f"{owner or 'another process'}; refusing to open another X5 writer"
                ) from exc
            raise

        os.ftruncate(fd, 0)
        metadata = (
            f"owner={self.owner}\n"
            f"pid={os.getpid()}\n"
            f"host={socket.gethostname()}\n"
            f"interface={self.interface}\n"
            f"acquired_wall_time={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
        )
        os.write(fd, metadata.encode("utf-8"))
        os.fsync(fd)
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        fd = self._fd
        self._fd = None
        try:
            os.ftruncate(fd, 0)
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> "CanOwnerLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def _read_lock_owner(fd: int) -> str:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        return os.read(fd, 4096).decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _safe_lock_name(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe) or "can"
