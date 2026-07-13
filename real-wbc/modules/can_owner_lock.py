from __future__ import annotations

import os

from modules.hardware_ownership import HardwareOwnershipLock


DEFAULT_CAN_LOCK_DIR = os.environ.get("GX_REAL_LOCK_DIR", "/run/lock/gx-real")


class CanOwnerLock(HardwareOwnershipLock):
    def __init__(
        self,
        interface: str,
        *,
        owner: str,
        lock_dir: str = DEFAULT_CAN_LOCK_DIR,
        real_hardware: bool = True,
    ):
        self.interface = str(interface)
        super().__init__(
            "x5-can",
            owner=owner,
            lock_dir=lock_dir,
            real_hardware=real_hardware,
        )
