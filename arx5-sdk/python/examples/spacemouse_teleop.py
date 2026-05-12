from queue import Queue
import os
import sys

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
from arx5_interface import (
    Arx5CartesianController,
    ControllerConfig,
    ControllerConfigFactory,
    EEFState,
    Gain,
    LogLevel,
    RobotConfigFactory,
)
from peripherals.spacemouse_shared_memory import Spacemouse
from multiprocessing.managers import SharedMemoryManager

import time
import click


LOG_LEVELS = {
    "trace": LogLevel.TRACE,
    "debug": LogLevel.DEBUG,
    "info": LogLevel.INFO,
    "warning": LogLevel.WARNING,
    "error": LogLevel.ERROR,
    "critical": LogLevel.CRITICAL,
    "off": LogLevel.OFF,
}


def start_teleop_recording(
    controller: Arx5CartesianController,
    ori_speed: float,
    pos_speed: float,
    gripper_speed: float,
    deadzone_threshold: float,
    window_size: int,
    cmd_dt: float,
    preview_time: float,
    workspace_xyz,
    workspace_rpy,
):

    # For earlier spacemouse versions (wired version), the readout might be not zero even after it is released
    # If you are using the wireless 3Dconnexion spacemouse, you can set the deadzone_threshold to 0.0 for better sensitivity
    home_pose_6d = controller.get_home_pose().copy()
    target_pose_6d = home_pose_6d.copy()
    if workspace_xyz is not None:
        workspace_xyz = np.asarray(workspace_xyz, dtype=float)
    if workspace_rpy is not None:
        workspace_rpy = np.asarray(workspace_rpy, dtype=float)

    target_gripper_pos = 0.0

    UPDATE_TRAJ = True
    # False: only override single points with position control
    # True: send a trajectory command every update_dt. Will include velocity. This is

    pose_x_min = target_pose_6d[0]
    spacemouse_queue = Queue(window_size)
    robot_config = controller.get_robot_config()

    avg_error = np.zeros(6)
    avg_cnt = 0
    prev_eef_cmd = EEFState()
    eef_cmd = EEFState()

    with SharedMemoryManager() as shm_manager:
        with Spacemouse(
            shm_manager=shm_manager, deadzone=deadzone_threshold, max_value=500
        ) as sm:

            def get_filtered_spacemouse_output(sm: Spacemouse):
                state = sm.get_motion_state_transformed()
                # Remove the deadzone and normalize the output
                positive_idx = state >= deadzone_threshold
                negative_idx = state <= -deadzone_threshold
                state[positive_idx] = (state[positive_idx] - deadzone_threshold) / (
                    1 - deadzone_threshold
                )
                state[negative_idx] = (state[negative_idx] + deadzone_threshold) / (
                    1 - deadzone_threshold
                )

                if (
                    spacemouse_queue.maxsize > 0
                    and spacemouse_queue._qsize() == spacemouse_queue.maxsize
                ):
                    spacemouse_queue._get()
                spacemouse_queue.put_nowait(state)
                return np.mean(np.array(list(spacemouse_queue.queue)), axis=0)

            print("Teleop tracking ready. Waiting for spacemouse movement to start.")

            while True:
                button_left = sm.is_button_pressed(0)
                button_right = sm.is_button_pressed(1)
                state = get_filtered_spacemouse_output(sm)
                if state.any() or button_left or button_right:
                    print(f"Start tracking!")
                    break
                eef_cmd = controller.get_eef_cmd()
                prev_eef_cmd = eef_cmd
            start_time = time.monotonic()
            loop_cnt = 0
            while True:

                print(
                    f"Time elapsed: {time.monotonic() - start_time:.03f}s",
                    end="\r",
                )
                # Spacemouse state is in the format of (x y z roll pitch yaw)
                state = get_filtered_spacemouse_output(sm)
                button_left = sm.is_button_pressed(0)
                button_right = sm.is_button_pressed(1)
                if button_left and button_right:
                    print(f"Avg 6D pose error: {avg_error / avg_cnt}")
                    # Traj with vel Avg 6D pose error:      [ 0.0004  0.0002 -0.0016  0.0002  0.0032  0.0005]
                    # Single point without vel:             [-0.0002 -0.0006 -0.0026  0.0027  0.0042 -0.0017]
                    # Traj without vel Avg 6D pose error:   [ 0.0005  0.0008 -0.005  -0.0024  0.0073 -0.0001]

                    controller.reset_to_home()
                    config = controller.get_robot_config()
                    home_pose_6d = controller.get_home_pose().copy()
                    target_pose_6d = home_pose_6d.copy()
                    target_gripper_pos = 0.0
                    loop_cnt = 0
                    start_time = time.monotonic()

                    continue
                elif button_left and not button_right:
                    gripper_cmd = 1
                elif button_right and not button_left:
                    gripper_cmd = -1
                else:
                    gripper_cmd = 0
                # print(state, target_gripper_pos)
                proposed_pose_6d = target_pose_6d.copy()
                proposed_pose_6d[:3] += state[:3] * pos_speed * cmd_dt
                proposed_pose_6d[3:] += state[3:] * ori_speed * cmd_dt
                if workspace_xyz is None:
                    target_pose_6d[:3] = proposed_pose_6d[:3]
                else:
                    target_pose_6d[:3] = np.clip(
                        proposed_pose_6d[:3],
                        home_pose_6d[:3] - workspace_xyz,
                        home_pose_6d[:3] + workspace_xyz,
                    )
                if workspace_rpy is None:
                    target_pose_6d[3:] = proposed_pose_6d[3:]
                else:
                    target_pose_6d[3:] = np.clip(
                        proposed_pose_6d[3:],
                        home_pose_6d[3:] - workspace_rpy,
                        home_pose_6d[3:] + workspace_rpy,
                    )
                target_gripper_pos += gripper_cmd * gripper_speed * cmd_dt
                if target_gripper_pos >= robot_config.gripper_width:
                    target_gripper_pos = robot_config.gripper_width
                elif target_gripper_pos <= 0:
                    target_gripper_pos = 0
                loop_cnt += 1
                while time.monotonic() < start_time + loop_cnt * cmd_dt:
                    pass
                current_timestamp = controller.get_timestamp()
                prev_eef_cmd = eef_cmd
                # if target_pose_6d[0] < pose_x_min:
                #     target_pose_6d[0] = pose_x_min
                eef_cmd.pose_6d()[:] = target_pose_6d
                eef_cmd.gripper_pos = target_gripper_pos
                eef_cmd.timestamp = current_timestamp + preview_time

                if UPDATE_TRAJ:
                    # This will calculate the velocity automatically
                    controller.set_eef_traj([eef_cmd])

                # Or sending single eef_cmd:
                else:
                    # Only position control
                    controller.set_eef_cmd(eef_cmd)

                output_eef_cmd = controller.get_eef_cmd()
                eef_state = controller.get_eef_state()
                avg_error += output_eef_cmd.pose_6d() - eef_state.pose_6d()
                avg_cnt += 1

                print(f"6DPose Error: {output_eef_cmd.pose_6d() - eef_state.pose_6d()}")


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
@click.option("--pos-speed", default=0.10, show_default=True, help="Max Cartesian translation speed in m/s.")
@click.option("--ori-speed", default=0.30, show_default=True, help="Max Cartesian rotation speed in rad/s.")
@click.option("--gripper-speed", default=0.03, show_default=True, help="Gripper command speed in m/s.")
@click.option("--deadzone", default=0.30, show_default=True, help="SpaceMouse normalized deadzone.")
@click.option("--window-size", default=8, show_default=True, help="Moving-average filter window.")
@click.option("--cmd-dt", default=0.02, show_default=True, help="Command loop period in seconds.")
@click.option("--preview-time", default=0.08, show_default=True, help="Trajectory preview time in seconds.")
@click.option(
    "--log-level",
    type=click.Choice(sorted(LOG_LEVELS.keys())),
    default="info",
    show_default=True,
    help="ARX5 SDK log verbosity.",
)
@click.option(
    "--workspace-xyz",
    nargs=3,
    type=float,
    default=None,
    help="XYZ limits around home pose in meters.",
)
@click.option(
    "--workspace-rpy",
    nargs=3,
    type=float,
    default=None,
    help="RPY limits around home pose in radians.",
)
def main(
    model: str,
    interface: str,
    pos_speed: float,
    ori_speed: float,
    gripper_speed: float,
    deadzone: float,
    window_size: int,
    cmd_dt: float,
    preview_time: float,
    log_level: str,
    workspace_xyz,
    workspace_rpy,
):

    robot_config = RobotConfigFactory.get_instance().get_config(model)
    sdk_root = os.path.dirname(ROOT_DIR)
    models_dir = os.environ.get("GX_REAL_ARX5_MODELS_DIR", os.path.join(sdk_root, "models"))
    urdf_path = os.path.join(models_dir, f"{model}.urdf")
    if os.path.isfile(urdf_path):
        robot_config.urdf_path = urdf_path
        print(f"Using ARX5 URDF: {robot_config.urdf_path}")
    else:
        print(f"Warning: ARX5 URDF not found at {urdf_path}; using SDK default.")
    controller_config = ControllerConfigFactory.get_instance().get_config(
        "cartesian_controller", robot_config.joint_dof
    )
    # controller_config.interpolation_method = "cubic"
    controller_config.default_kp = controller_config.default_kp
    controller = Arx5CartesianController(robot_config, controller_config, interface)
    controller.reset_to_home()

    robot_config = controller.get_robot_config()
    gain = Gain(robot_config.joint_dof)
    controller.set_log_level(LOG_LEVELS[log_level])
    np.set_printoptions(precision=4, suppress=True)
    workspace_xyz_text = "system" if workspace_xyz is None else workspace_xyz
    workspace_rpy_text = "system" if workspace_rpy is None else workspace_rpy
    print(
        "SpaceMouse settings: "
        f"pos_speed={pos_speed}, ori_speed={ori_speed}, deadzone={deadzone}, "
        f"workspace_xyz={workspace_xyz_text}, workspace_rpy={workspace_rpy_text}, "
        f"log_level={log_level}"
    )
    try:
        start_teleop_recording(
            controller,
            ori_speed=ori_speed,
            pos_speed=pos_speed,
            gripper_speed=gripper_speed,
            deadzone_threshold=deadzone,
            window_size=window_size,
            cmd_dt=cmd_dt,
            preview_time=preview_time,
            workspace_xyz=workspace_xyz,
            workspace_rpy=workspace_rpy,
        )
    except KeyboardInterrupt:
        print(f"Teleop recording is terminated. Resetting to home.")
        controller.reset_to_home()
        controller.set_to_damping()


if __name__ == "__main__":
    main()
