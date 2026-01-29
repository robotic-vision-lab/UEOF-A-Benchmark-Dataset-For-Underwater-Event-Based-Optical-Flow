from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def process_and_save_splits():
    pose_csv_path = "camM0_poses_transformation_matrix.csv"
    imu_csv_path = "imu0_data.csv"

    splits = {
        "s1": (0, 450),
        "s2": (451, 1373),
        "s3": (1374, 2745),
        "s4": (2746, 3011),
        "s5": (3012, 4713),
    }

    print("Loading data...")
    # Camera Poses
    cam_df = pd.read_csv(pose_csv_path)
    ts_cam_ns = cam_df["# timestamp [ns]"].values
    ts_cam_s = ts_cam_ns * 1e-9

    # Extract Position (World Frame)
    p_wc = cam_df[["s_x_vw", "s_y_vw", "s_z_vw"]].values

    # Extract Rotation (Camera to World) - reshape to (N, 3, 3)
    r_cols = ["r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"]
    R_wc = cam_df[r_cols].values.reshape(-1, 3, 3)

    # IMU Data
    imu_df = pd.read_csv(imu_csv_path)
    ts_imu_s = imu_df["# timestamp [ns]"].values * 1e-9
    w_imu = imu_df[
        ["w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]"]
    ].values

    print("2. Computing velocities (Full Sequence)...")

    # Interpolate IMU to Camera Timestamps
    imu_interp = interp1d(
        ts_imu_s, w_imu, axis=0, kind="linear", fill_value="extrapolate"
    )

    lin_vels_list = []
    ang_vels_list = []
    ts_us_list = []

    num_poses = len(ts_cam_s)

    # Calculate velocities for intervals i -> i+1
    for i in range(num_poses - 1):
        t_curr = ts_cam_s[i + 1]
        dt = t_curr - ts_cam_s[i]

        if dt <= 0:
            # potential duplicate/bad timestamps
            lin_vels_list.append([0, 0, 0])
            ang_vels_list.append([0, 0, 0])
            ts_us_list.append(int(t_curr * 1e6))
            continue

        # Linear Velocity in World Frame
        v_world = (p_wc[i + 1] - p_wc[i]) / dt

        # Transform to Camera Frame: v_cam = R^T * v_world
        # Using Rotation at t_{i+1}
        v_cam = R_wc[i + 1].T @ v_world
        lin_vels_list.append(v_cam)

        # Angular Velocity (Sampled at t_{i+1})
        w_curr = imu_interp(t_curr)
        ang_vels_list.append(w_curr)

        # Timestamp (microseconds)
        ts_us_list.append(int(ts_cam_ns[i + 1] // 1000))

    full_lin_vels = np.array(lin_vels_list)
    full_ang_vels = np.array(ang_vels_list)
    full_ts_us = np.array(ts_us_list)

    print(f"Computed {len(full_ts_us)} velocity samples.")

    for name, (start_idx, end_idx) in splits.items():
        # add 1 to end_idx because Python slicing is exclusive at the upper bound
        # clamp to the actual length of the data
        s_start = max(0, start_idx)
        s_end = min(len(full_ts_us), end_idx + 1)

        # Extract slices
        s_lin = full_lin_vels[s_start:s_end]
        s_ang = full_ang_vels[s_start:s_end]
        s_ts = full_ts_us[s_start:s_end]

        filename = f"{name}_velocities.npz"

        np.savez(filename, lin_vels=s_lin, ang_vels=s_ang, ts_us=s_ts)

        print(
            f"Saved '{filename}': Frames {s_start}-{end_idx} ({len(s_ts)} samples)"
        )


if __name__ == "__main__":
    process_and_save_splits()
