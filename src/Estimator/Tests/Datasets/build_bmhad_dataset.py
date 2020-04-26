from glob import glob
import os
from subprocess import call

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
            joint_pos = (data[joint + "X"][frame_idx], data[joint + "Y"][frame_idx], data[joint + "Z"][frame_idx])
            pose_3d.append(joint_pos)
        full_pose_3d.append(pose_3d)

    return np.array(full_pose_3d) * 10


def bmhad_to_ours(joints):
    num_frames = joints.shape[0]
    new_joints = np.zeros((num_frames, 12, joints.shape[2]))
    head_sizes = np.zeros((num_frames))

    for idx, current_joints in enumerate(joints):
        new_joints[idx, 0] = np.mean(current_joints[:2], axis=0).astype(joints.dtype)  # head
        new_joints[idx, 1:7] = current_joints[2:8]
        new_joints[idx, 7:9] = current_joints[9:11]
        new_joints[idx, 9:11] = current_joints[12:14]
        new_joints[idx, 11] = np.mean(current_joints[np.array([8, 11])], axis=0).astype(joints.dtype)  # pelvis
        head_sizes[idx] = np.linalg.norm(current_joints[0] - current_joints[1])

    return new_joints, head_sizes


def get_cam_params(cam):
    """
    Get camera parameters
    :param cam: str, cam ID
    :return: dict, np.array - camera parameters, camera world coordinates
    """
    # KINECT params
    K = {}
    if cam == "k02":
        K["R"] = np.reshape([-0.798016667, - 0.041981064, 0.601171315, -0.059102636, -0.987309396, -0.147400886, 0.599730134, -0.153159171, 0.785408199], (3, 3)).T.astype(np.float64)
        K["t"] = np.array([[26.147224426, 853.124328613, 2533.297607422]]).T.astype(np.float64)
        K["K"] = np.reshape([532.33691406, 0., 323.22338867, 0., 532.80218506, 265.27493286, 0., 0., 1.], (3, 3)).astype(np.float64)
        K["d"] = np.array([0.18276334, -0.35502717, -6.75550546e-004, -9.90863307e-004]).T.astype(np.float64)
    elif cam == "k01":
        K["R"] = np.reshape([0.869593024, 0.005134047, -0.493742496, 0.083783410, -0.986979902, 0.137298822, -0.486609042, -0.160761520, -0.858700991], (3, 3)).T.astype(np.float64)
        K["t"] = np.array([[-844.523864746, 763.838439941, 3232.193359375]]).T.astype(np.float64)
        K["K"] = np.reshape([531.49230957, 0., 314.63775635, 0., 532.39190674, 252.53335571, 0., 0., 1.], (3, 3)).astype(np.float64)
        K["d"] = np.array([0.19607373, -0.36734107, -2.47962005e-003, -1.89774996e-003]).T.astype(np.float64)
    else:
        raise Exception("Unknown cam: %s" % cam)

    K["T"] = np.append(np.append(K["R"].T,  K["t"], axis=1), np.array([[0, 0, 0, 1]]).astype(np.float64), axis=0).astype(np.float64)

    return K

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
    rvec = cv2.Rodrigues(K["R"].T)[0]

    joints_2d, _ = cv2.projectPoints(new_joints, rvec, K["t"], K["K"], K["d"])
    joints_3d = np.zeros_like(joints)
    for idx, joint in enumerate(new_joints):
        joints_3d[idx] = np.dot(K["T"], np.append(joint, [1]).T)[:3]

    return np.squeeze(joints_2d), joints_3d


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

    joint_order_fancier = ["head"]
    joint_order_fancier += ["right_shoulder", "right_elbow", "right_hand"]
    joint_order_fancier += ["left_shoulder", "left_elbow", "left_hand"]
    joint_order_fancier += ["right_knee", "right_foot"]
    joint_order_fancier += ["left_knee", "left_foot"]
    joint_order_fancier += ["pelvis"]

    scene_dir = os.path.join(data_dir, scene_code)

    f = h5py.File(fname, mode="w")
    f.attrs["joint_order"] = joint_order_fancier

    # Calibration
    calibration_dtype = np.dtype([
        ("K", np.float32, (3, 3)),
        ("R", np.float32, (3, 3)),
        ("t", np.float32, (3)),
        ("dist", np.float32, (4)),
        ("resolution", np.uint16, (2)),
    ])

    cam_id = "k" + scene_code.split("k")[-1]
    cam_cfg_fname = os.path.join(data_dir, "camcfg_%s.yml" % cam_id)
    cam_cfg = parse_cam_cfg(cam_cfg_fname)

    # Pose and image data
    corr_frames = np.loadtxt(glob(os.path.join(scene_dir, "corr*.txt"))[0])
    num_frames = corr_frames.shape[0]

    im_height, im_width = cam_cfg["resolution"]
    pose_dtype = np.dtype([
        ("camera_id", np.str_, 20),
        ("subject_id", np.str_, 20),
        ("action_id", np.str_, 20),
        ("repetition_id", np.str_, 20),
        ("rgb_video", np.uint8, (num_frames, im_height, im_width, 3)),
        ("depth_video", np.uint16, (num_frames, im_height, im_width)),
        ("pose_2d", np.float32, (num_frames, 12, 2)),
        ("pose_3d", np.float32, (num_frames, 12, 3)),
        ("pose_3d_world", np.float32, (num_frames, 12, 3)),
        ("head_sizes_2d", np.float32, (num_frames)),
        ("head_sizes_3d", np.float32, (num_frames)),
    ])
    f.create_dataset("calibration", (1,), calibration_dtype)

    f.create_dataset("pose", (1,), pose_dtype)

    video_frames = corr_frames[:, 0]
    rgb, depth = parse_video(os.path.join(scene_dir, "video"), video_frames)

    mocap_frames = corr_frames[:, 2]

    call(["bvh-converter", glob(os.path.join(scene_dir, "*.bvh"))[0]])
    joints_3d_world = parse_mocap(glob(os.path.join(scene_dir, "*.csv"))[0], joint_order, mocap_frames)

    K = get_cam_params(cam_id)

    joints_3d_aux = joints_3d_world.copy()
    joints_3d_world[:, :, 0] = joints_3d_aux[:, :, 2]
    joints_3d_world[:, :, 1] = joints_3d_aux[:, :, 0]
    joints_3d_world[:, :, 2] = joints_3d_aux[:, :, 1]
    joints_2d = np.zeros((joints_3d_world.shape[0], joints_3d_world.shape[1], 2))
    joints_3d_camera = np.zeros_like(joints_3d_world)
    for frame_idx, skel in enumerate(joints_3d_world):
        joints_2d[frame_idx], joints_3d_camera[frame_idx] = project_points(K, skel)

    joints_3d_world, _ = bmhad_to_ours(joints_3d_world)
    joints_3d_camera, head_sizes_3d = bmhad_to_ours(joints_3d_camera)
    joints_2d, head_sizes_2d = bmhad_to_ours(joints_2d)

    if debug:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection='3d')

        def animate(i):
            ax1.clear()
            ax1.scatter(joints_2d[i, :, 0], joints_2d[i, :, 1], s=30)
            for idx, joint_2d in enumerate(joints_2d[i]):
                ax1.text(joint_2d[0] + 5, joint_2d[1] + 5, str(idx), color="red", fontsize=9)
            ax1.imshow(rgb[i])

            ax2.clear()
            ax2.scatter(joints_3d_camera[i, :, 2], joints_3d_camera[i, :, 0], -joints_3d_camera[i, :, 1])
            ax2.set_xlim(2000, 4000)
            ax2.set_ylim(-1500, 500)
            ax2.set_zlim(-1000, 1000)
            ax2.view_init(30, i * 2)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani = animation.FuncAnimation(fig, animate, interval=1, save_count=num_frames)
        ani.save(os.path.join(data_dir, '%s.mp4' % scene_code), writer=writer)

    calibration = np.array((K["K"], K["R"], np.squeeze(K["t"]), K["d"], cam_cfg["resolution"]), dtype=calibration_dtype)
    f["calibration"][0] = calibration

    pose = np.array((
        cam_id,
        "s" + scene_code.split("s")[-1][:2],
        "a" + scene_code.split("a")[-1][:2],
        "r" + scene_code.split("r")[-1][:2],
        rgb,
        depth,
        joints_2d,
        joints_3d_camera,
        joints_3d_world,
        head_sizes_2d,
        head_sizes_3d,
    ), dtype=pose_dtype)
    f["pose"][0] = pose

    return f


if __name__ == "__main__":
    data_dir = "/home/dpascualhe/repos/2017-tfm-david-pascual/src/Estimator/Tests/Datasets/BMHAD"
    for scene_dir in glob(data_dir + "/*r02*/"):
        scene_code = scene_dir.split("/")[-2]
        print("Parsing %s" % scene_code)
        fname = os.path.join(data_dir, "hdf5/bmhad_%s.h5" % scene_code)
        dataset = build_dataset(fname, data_dir, scene_code=scene_code, debug=True)
        dataset.close()
