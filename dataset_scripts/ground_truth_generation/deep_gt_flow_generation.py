import os
import struct
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def write_flo(file, flow):
    # flow: HxWx2 float32
    with open(file, "wb") as f:
        f.write(b"PIEH")
        h, w, _ = flow.shape
        f.write(struct.pack("ii", w, h))
        flow.astype(np.float32).tofile(f)


def load_intrinsics(yaml_file):
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    try:
        width = data["resolution"]["width"]
        height = data["resolution"]["height"]
    except (KeyError, TypeError):
        width = data["width"]
        height = data["height"]

    fx = data["focal_length"] / data["sensor_width"] * width
    fy = data["focal_length"] / data["sensor_height"] * height
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K


def load_poses(csv_file):
    df = pd.read_csv(csv_file, header=0)
    poses = []
    for _, row in df.iterrows():
        # Cols: ts, r11,r12,r13, tx, r21,r22,r23, ty, r31,r32,r33, tz
        rot_flat = np.array(
            [
                row["r11"],
                row["r12"],
                row["r13"],
                row["r21"],
                row["r22"],
                row["r23"],
                row["r31"],
                row["r32"],
                row["r33"],
            ]
        )
        trans = np.array([row["s_x_vw"], row["s_y_vw"], row["s_z_vw"]])

        T = np.eye(4)
        T[:3, :3] = rot_flat.reshape(3, 3)
        T[:3, 3] = trans
        poses.append(T)
    return poses, df.iloc[:, 0].values


def backproject(depth, K):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3).T
    z = depth.reshape(-1)
    Kinv = np.linalg.inv(K)
    pts_cam = (Kinv @ uv1) * z
    return pts_cam.T  # (N,3)


def project(points_3d, K):
    pts_proj = (K @ points_3d.T).T
    uv = pts_proj[:, :2] / pts_proj[:, 2:3]
    return uv


def bilinear_interpolate(img, x, y):
    # x, y: arrays of float coordinates
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, img.shape[1] - 1)
    x1 = np.clip(x1, 0, img.shape[1] - 1)
    y0 = np.clip(y0, 0, img.shape[0] - 1)
    y1 = np.clip(y1, 0, img.shape[0] - 1)

    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def generate_flow(
    rgb_dir,
    depth_dir,
    pose_csv,
    cam_yaml,
    timestamps_csv,
    out_dir,
    d_start=0.0,
    d_end=25.0,
    epsilon=0.01,
):
    os.makedirs(out_dir, exist_ok=True)

    K = load_intrinsics(cam_yaml)
    poses, pose_ts = load_poses(pose_csv)

    frames = sorted(Path(rgb_dir).glob("*.png"))
    depths = sorted(Path(depth_dir).glob("*.png"))

    assert len(frames) == len(depths), "RGB and Depth counts differ"

    # Load frame timestamps
    if timestamps_csv:
        ts_df = pd.read_csv(timestamps_csv, header=0)
        frame_ts_ns = ts_df.iloc[:, 0].values  # Col 0: timestamp [ns]
        # 10Hz
        frame_ts = np.arange(len(frame_ts_ns)) * 100000000
    else:
        # Fallback
        frame_indices = np.array([int(f.stem.split("-")[-1]) for f in frames])
        frame_ts = (frame_indices - frame_indices[0]) * 100000000

    print(f"Processing {len(frames)-1} frame pairs...")

    # Subsections
    subseqs = {
        "s1": (0, 450),
        "s2": (451, 1373),
        "s3": (1374, 2745),
        "s4": (2746, 3011),
        "s5": (3012, 4713),
    }

    for sub_name, (start_i, end_i) in subseqs.items():
        sub_out_dir = Path(out_dir) / sub_name
        os.makedirs(sub_out_dir, exist_ok=True)
        print(
            f"Generating flows for {sub_name}: frames {start_i}-{end_i} (flows {start_i}-{end_i-1})"
        )

        for i in tqdm(range(start_i, end_i)):
            f0 = frames[i]
            f1 = frames[i + 1] if i + 1 < len(frames) else None
            d0_file = depths[i]
            d1_file = depths[i + 1] if i + 1 < len(frames) else None

            # Load depth as meters (16-bit PNG -> [0,1] * range)
            d0_raw = cv2.imread(str(d0_file), cv2.IMREAD_UNCHANGED).astype(
                np.float32
            )
            depth_m = d_start + (d0_raw / 65535.0) * (d_end - d_start)

            d1_raw = cv2.imread(str(d1_file), cv2.IMREAD_UNCHANGED).astype(
                np.float32
            )
            depth_m1 = d_start + (d1_raw / 65535.0) * (d_end - d_start)

            # Sync poses via timestamps
            ts0 = frame_ts[i]
            ts1 = frame_ts[i + 1] if i + 1 < len(frames) else frame_ts[-1]
            idx0 = np.argmin(np.abs(pose_ts - ts0))
            idx1 = np.argmin(np.abs(pose_ts - ts1))

            if abs(pose_ts[idx0] - ts0) > 1e6:  # >1ms warning
                print(
                    f"Warning: Frame {i} ts mismatch: {ts0} vs {pose_ts[idx0]} ns"
                )

            T_w_c0 = poses[idx0]
            T_w_c1 = poses[idx1]
            T_c1_w = np.linalg.inv(T_w_c1)

            # Backproject pixels at t0
            pts_c0 = backproject(depth_m, K)  # (N,3)
            pts_h = np.hstack([pts_c0, np.ones((pts_c0.shape[0], 1))])  # (N,4)

            # World transform
            pts_w = (T_w_c0 @ pts_h.T).T  # (N,4)
            pts_c1_h = (T_c1_w @ pts_w.T).T  # (N,4)
            pts_c1 = pts_c1_h[:, :3]  # (N,3)

            # Project
            h, w = depth_m.shape
            uv0 = (
                np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1)
                .reshape(-1, 2)
                .astype(np.float32)
            )
            uv1 = project(pts_c1, K)

            # Masks for flow computation (compute displacements where possible)
            valid_depth = (depth_m > 1e-3).flatten()
            z_positive = pts_c1[:, 2] > 0
            valid_flow = valid_depth & z_positive

            # Additional masks for validity (in_bounds and not occluded)
            in_bounds = (
                (uv1[:, 0] >= 0)
                & (uv1[:, 0] < w)
                & (uv1[:, 1] >= 0)
                & (uv1[:, 1] < h)
            )

            # Interpolate depth at projected points for occlusion check
            interpolated_d1 = np.full((uv1.shape[0],), np.nan, dtype=np.float32)
            interp_mask = in_bounds
            interpolated_d1[interp_mask] = bilinear_interpolate(
                depth_m1, uv1[interp_mask, 0], uv1[interp_mask, 1]
            )

            occluded = pts_c1[:, 2] > interpolated_d1 + epsilon

            valid_mask = valid_flow & in_bounds & ~occluded

            # Flow (compute for all valid_flow, even if out of bounds or occluded)
            flow = np.zeros((h * w, 2), dtype=np.float32)
            flow[valid_flow] = uv1[valid_flow] - uv0[valid_flow]
            flow_img = flow.reshape(h, w, 2)

            # Validity mask image (255 for valid, 0 otherwise)
            mask = np.zeros((h * w), dtype=np.uint8)
            mask[valid_mask] = 255
            mask_img = mask.reshape(h, w)

            out_file = sub_out_dir / f"{i:05d}.flo"
            write_flo(out_file, flow_img)

            out_mask = sub_out_dir / f"{i:05d}.png"
            cv2.imwrite(str(out_mask), mask_img)

            if i == start_i:
                print(
                    f"{sub_name} first flow stats: valid={np.sum(valid_mask)/(h*w)*100:.1f}%, max={np.max(np.abs(flow)): .2f} px"
                )
                print(f"{sub_name} T0 origin in world: {T_w_c0[:3,3]}")


if __name__ == "__main__":
    root_dir = Path("/data/VAROS/2021-08-17_SEQ1/2021-08-17_SEQ1/vehicle0/cam0")
    rgb_dir = root_dir / "A"
    depth_dir = root_dir / "D"
    pose_csv = root_dir / "camM0_poses/camM0_poses_transformation_matrix.csv"
    cam_yaml = root_dir / "camM0.yaml"
    timestamps_csv = root_dir / "camM0_timestamps.csv"
    out_dir = "./output/"

    generate_flow(
        rgb_dir, depth_dir, pose_csv, cam_yaml, timestamps_csv, out_dir
    )
