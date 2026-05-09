uint32 tick
float64 system_time

# Incremental EEF command in arm frame: dx, dy, dz, droll, dpitch, dyaw.
float64[3] translation
float64[3] rotation_rpy

bool hold
