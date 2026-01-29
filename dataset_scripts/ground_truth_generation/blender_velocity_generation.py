import bpy
import numpy as np

camera_name = "Camera"
output_filename = bpy.path.abspath("//camera_vels.npz")

scene = bpy.context.scene
cam = bpy.data.objects.get(camera_name)
if cam is None:
    raise ValueError(f"Camera object named '{camera_name}' not found")


# convert 4x4 matrix to (R, t)
def decompose_matrix(mat4):
    # mat4 is a mathutils.Matrix 4x4, camera to world
    R = mat4.to_3x3()
    t = mat4.translation
    return R, t


# rotation matrix logarithm -> rotation vector (axis angle form)
def log_so3(R):
    # R is a 3x3 rotation (as numpy array)
    # compute angle
    trace = np.trace(R)
    theta = np.arccos(np.clip((trace - 1) / 2.0, -1.0, 1.0))
    if abs(theta) < 1e-6:
        return np.zeros(3)
    # compute axis
    # (R - R^T) vector form
    omega_mat = (R - R.T) / (2 * np.sin(theta))
    wx = omega_mat[2, 1]
    wy = omega_mat[0, 2]
    wz = omega_mat[1, 0]
    axis = np.array([wx, wy, wz])
    return axis * theta


rots = []
trans = []
timestamps = []

# Blender frames -> microseconds
fps_total = scene.render.fps * scene.render.fps_base
dt = 1.0 / fps_total
frame_start = scene.frame_start
frame_end = scene.frame_end

for frame in range(frame_start, frame_end + 1):
    scene.frame_set(frame)
    # Get camera to world 4x4
    mw = cam.matrix_world.copy()
    R_world_cam, t_world_cam = decompose_matrix(mw)
    R_np = np.array(R_world_cam)  # shape (3,3)
    t_np = np.array([t_world_cam.x, t_world_cam.y, t_world_cam.z])
    rots.append(R_np)
    trans.append(t_np)

    # timestamp in microseconds
    # tframe = (frame - frame_start) * dt seconds
    t_us = int((frame - frame_start) * dt * 1e6)
    timestamps.append(t_us)

# compute velocities (N = number of frames minus 1)
N = len(trans) - 1
lin_vels = np.zeros((N, 3), dtype=np.float64)
ang_vels = np.zeros((N, 3), dtype=np.float64)

for i in range(N):
    t1 = trans[i]
    t2 = trans[i + 1]
    R1 = rots[i]
    R2 = rots[i + 1]

    # Linear velocity in camera frame at time i:
    # v = R1^T (t2 - t1) / dt
    v_world = (t2 - t1) / dt
    v_cam = R1.T @ v_world
    lin_vels[i] = v_cam

    # Angular velocity - use rotation delta
    dR = R1.T @ R2
    rotvec = log_so3(dR)
    w_cam = rotvec / dt
    ang_vels[i] = w_cam

# Flip Y/Z to match rectified camera frame (X right, Y down, Z forward)
lin_vels[:, 1] *= -1  # Y: Blender up → down
lin_vels[:, 2] *= -1  # Z: Blender backward → forward
ang_vels[:, 1] *= -1
ang_vels[:, 2] *= -1

ts_us = np.array(timestamps[1:], dtype=np.int64)

np.savez_compressed(
    output_filename, lin_vels=lin_vels, ang_vels=ang_vels, ts_us=ts_us
)

print(f"Exported velocities: {lin_vels.shape[0]} samples")
print("Saved .npz to:", output_filename)
