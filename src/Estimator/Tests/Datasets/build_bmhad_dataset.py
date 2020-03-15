from glob import glob
import os

import cv2
import h5py
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def parse_cam_cfg(fname):
    """
    Parse camera configuration YAML file generated w/ OpenCV
    :param fname: str, camera configuration file name
    :return: dict, camera configuration
    """
    f = cv2.FileStorage(fname, cv2.FILE_STORAGE_READ)
    cam_cfg = {}
    cam_cfg["K"] = f.getNode("Camera_1").getNode("K").mat().astype(np.float32)
    cam_cfg["dist"] = f.getNode("Camera_1").getNode("Dist").mat().astype(np.float32)
    cam_cfg["resolution"] = (
        int(f.getNode("Camera_1").getNode("imgHeight").real()),
        int(f.getNode("Camera_1").getNode("imgWidth").real()))

    return cam_cfg


def parse_video(video_dir, frames_indices):
    """
    Read RGB and depth video from a given folder
    :param video_dir: str, directory with RGB + depth frames
    :return: np.arrays, depth and rgb videos
    """
    rgb = []
    depth = []
    for frame_idx in frames_indices:
        frame_idx = int(frame_idx)
        rgb_frame = glob(os.path.join(video_dir, "*color_%05d*" % frame_idx))
        depth_frame = glob(os.path.join(video_dir, "*depth_%05d*" % frame_idx))
        rgb.append(cv2.imread(rgb_frame[0], -1)[:, :, ::-1])
        depth.append(cv2.imread(depth_frame[0], -1))

    return np.array(rgb), np.array(depth)


def parse_mocap(fname, joints_names, frames_indices):
    """
    Parse mocap data in a given BVH file
    :param fname: str, bvh file name
    :param joints_names: list, required joints names
    :param num_frames_in_video: int, number of frames in video
    :return: np.arrays, 2d and 3d joints
    """

    data = np.genfromtxt(fname, dtype=float, delimiter=',', names=True)
    full_pose_3d = []

    for frame_idx in frames_indices:
        frame_idx = int(frame_idx) + 1
        pose_3d = []
        for joint in joints_names:
            pose_3d.append((data[joint + "X"][frame_idx], data[joint + "Y"][frame_idx], data[joint + "Z"][frame_idx]))
        full_pose_3d.append(pose_3d)

    return np.array(full_pose_3d) * 10


def get_cam_world_coords():
    """
    Generate camera world coordinates
    :return: dict, np.array - camera parameters, camera world coordinates
    """
    # KINECT params
    K = {}
    K["R"] = np.reshape([-0.798016667, - 0.041981064, 0.601171315, -0.059102636, -0.987309396, -0.147400886, 0.599730134, -0.153159171, 0.785408199], (3, 3)).T
    K["t"] = np.array([[26.147224426, 853.124328613, 2533.297607422]]).T
    K["T"] = np.append(np.append(K["R"],  K["t"], axis=1), np.array([[0, 0, 0, 1]]), axis=0)
    K["Tinv"] = np.append(np.append(K["R"].T, np.dot(-K["R"].T, K["t"]), axis=1), np.array([[0, 0, 0, 1]]), axis=0)
    K["K"] = np.reshape([532.33691406, 0., 323.22338867, 0., 532.80218506, 265.27493286, 0., 0., 1.], (3, 3))
    K["d"] = np.array([0.18276334, -0.35502717, -6.75550546e-004, -9.90863307e-004]).T

    # compute camera position in the woorld coordinate frame
    z = np.array([0, 0, 0, 1]).T
    K["p"] = np.dot(K["Tinv"], z)

    return K, K["p"][:3]

def project_points(K, joints):
    """
    Project world coordinates into image plane given camera parameters.
    :param K: dict, camera parameters
    :param joints: np.array, world coordinates
    :return: np.array, image plane coordinates
    """
    # Homogeneous transformation matrix from world coordinate frame to camera coordinate frame
    joints = np.float32(joints).reshape(-1,3)
    new_joints = np.zeros_like(joints)
    new_joints[:, 0] = joints[:, 1]
    new_joints[:, 1] = joints[:, 2]
    new_joints[:, 2] = joints[:, 0]
    rvec = cv2.Rodrigues(K["R"])[0]

    joints_2d, _ = cv2.projectPoints(new_joints, rvec, K["t"], K["K"], K["d"])  #, cv2.Rodrigues(K["R"]), K["t"], K["K"], K["d"])
    return np.squeeze(joints_2d)


def build_dataset(fname, data_dir, scene_code, debug=False):
    """
    Build a HDF5 dataset for the given subjects, actions, repetitions and cameras
    :param fname: str, resulting HDF5 file name
    :param data_dir: str, path to data folder
    :param scene_code: str, code for scene to be included in the dataset (subjectIDactionIDrepetitionID)
    :return: HDF5 object, resulting dataset
    """
    joint_order = ["HeadEnd", "Head"]
    joint_order += ["RightArm", "RightForeArm", "RightHand"]
    joint_order += ["LeftArm", "LeftForeArm", "LeftHand"]
    joint_order += ["RightUpLeg", "RightLeg", "RightFoot"]
    joint_order += ["LeftUpLeg", "LeftLeg", "LeftFoot"]

    scene_dir = os.path.join(data_dir, scene_code)

    f = h5py.File(fname, mode="w")
    f.attrs["joint_order"] = joint_order

    # Calibration
    calibration_dtype = np.dtype([
        ("K", np.float32, (3, 3)),
        ("R", np.float32, (3, 3)),
        ("t", np.float32, (3)),
        ("dist", np.float32, (4)),
        ("resolution", np.uint16, (2)),
    ])

    f.create_dataset("calibration", (1,), calibration_dtype)

    cam_cfg_fname = glob(os.path.join(data_dir, "camcfg*.yml"))[0]
    cam_cfg = parse_cam_cfg(cam_cfg_fname)

    # Pose and image data
    corr_frames = np.loadtxt(glob(os.path.join(scene_dir, "corr*.txt"))[0])
    num_frames = corr_frames.shape[0]

    im_height, im_width = cam_cfg["resolution"]
    pose_dtype = np.dtype([
        ("subject_id", np.str_, 20),
        ("action_id", np.str_, 20),
        ("repetition_id", np.str_, 20),
        ("rgb_video", np.uint8, (num_frames, im_height, im_width, 3)),
        ("depth_video", np.uint16, (num_frames, im_height, im_width)),
        ("pose_2d", np.float32, (num_frames, 14, 2)),
        ("pose_3d", np.float32, (num_frames, 14, 3)),
    ])

    f.create_dataset("pose", (1,), pose_dtype)

    video_frames = corr_frames[:, 0]
    rgb, depth = parse_video(os.path.join(scene_dir, "video"), video_frames)

    mocap_frames = corr_frames[:, 2]
    joints_3d = parse_mocap(glob(os.path.join(scene_dir, "*.csv"))[0], joint_order, mocap_frames)

    K, cam_coords = get_cam_world_coords()

    joints_3d_aux = joints_3d.copy()
    joints_3d[:, :, 0] = joints_3d_aux[:, :, 2]
    joints_3d[:, :, 1] = joints_3d_aux[:, :, 0]
    joints_3d[:, :, 2] = joints_3d_aux[:, :, 1]
    joints_2d = np.zeros((joints_3d.shape[0], joints_3d.shape[1], 2))
    for frame_idx, skel in enumerate(joints_3d):
        joints_2d[frame_idx] = project_points(K, skel)

    if debug:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection='3d')
        def animate(i):
            ax1.clear()
            ax1.scatter(joints_2d[i, :, 0], joints_2d[i, :, 1], s=50)
            for idx, joint_2d in enumerate(joints_2d[i]):
                ax1.text(joint_2d[0] + 5, joint_2d[1] + 5, str(idx), color="red", fontsize=11)
            ax1.imshow(rgb[i])

            ax2.clear()
            ax2.scatter(joints_3d[i, :, 0], joints_3d[i, :, 1], joints_3d[i, :, 2])
            ax2.scatter(cam_coords[2], cam_coords[0], cam_coords[1], color="red")
            ax2.set_xlim(-2000, 2000)
            ax2.set_ylim(-2000, 2000)
            ax2.set_zlim(0, 2000)

        ani = animation.FuncAnimation(fig, animate, interval=1)
        plt.show()

    f["calibration"][0]["K"] = K["K"]
    f["calibration"][0]["R"] = K["R"]
    f["calibration"][0]["t"] = np.squeeze(K["t"])
    f["calibration"][0]["dist"] = K["d"]
    f["calibration"][0]["resolution"] = cam_cfg["resolution"]

    f["pose"][0]["subject_id"] = str(scene_code.split("s")[:2])
    f["pose"][0]["action_id"] = str(scene_code.split("a")[:2])
    f["pose"][0]["repetition_id"] = str(scene_code.split("r")[:2])
    f["pose"][0]["rgb_video"] = rgb
    f["pose"][0]["depth_video"] = depth
    f["pose"][0]["pose_2d"] = joints_2d
    f["pose"][0]["pose_3d"] = joints_3d

    return f


if __name__ == "__main__":
    fname = "bmhad_k02s01a01r01.h5"
    data_dir = "BMHAD"

    dataset = build_dataset(fname, data_dir, scene_code="s01a01r01", debug=False)
    dataset.close()
