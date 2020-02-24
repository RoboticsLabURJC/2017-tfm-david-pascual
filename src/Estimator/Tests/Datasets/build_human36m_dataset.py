from glob import glob
import os
import random

os.environ["CDF_LIB"] = "/home/dpascualhe/repos/3d-pose-eval/cdf37_0-dist/lib"

import cv2
import h5py
from matplotlib import pyplot as plt
import numpy as np
from spacepy import pycdf

SORTED_JOINTS_IDX = np.array([15, 13, 17, 18, 19, 25, 26, 27, 6, 7, 8, 1, 2, 3])
SORTED_JOINTS_NAMES = np.array(
    ["head", "neck", "lshoulder", "lelbow", "lwrist", "rshoulder", "relboow", "rwrist", "lhip", "lnkee", "lankle",
     "rhip", "rknee", "rankle"])


def retrieve_samples_loc(path, num_subjects, num_scenes_per_subject, cam_id):
    """
    Randomly retrieve samples of continuous video data.
    :param path: string, path to data.
    :param num_subjects: int, number of subjects included.
    :param num_scenes_per_subject: int, number of scenes included.
    :param cam_id: string, camera ID.
    :return: dict, samples retrieved w/ labels.
    """
    subjects_dirs = random.sample(list(glob(os.path.join(path, "s*"))), num_subjects)

    samples_location = []
    for subject_dir in subjects_dirs:
        valid_videos = glob(os.path.join(subject_dir, "Videos", "*%s*" % cam_id))
        videos = random.sample(valid_videos, num_scenes_per_subject)
        for video in videos:
            scene = video.split(".")[-3].split("/")[-1]
            sample = {
                "subject": subject_dir.split("s")[-1],
                "scene": scene,
                "cam": cam_id,
                "video": video,
                "tof": glob(os.path.join(subject_dir, "TOF", "%s.cdf" % scene))[0],
                "pose_video": glob(os.path.join(subject_dir, "D2_Positions", "%s.%s.cdf" % (scene, cam_id)))[0],
                "pose_3d": glob(os.path.join(subject_dir, "D3_Positions_mono", "%s.%s.cdf" % (scene, cam_id)))[0]
            }
            samples_location.append(sample)
    return samples_location


def get_tof_coords(pose_3d, calib_matrix):
    """
    Return pixel coords in TOF image given a pose 3d and a calibration matrix.
    :param pose_3d: np.array, 3d human pose.
    :param calib_matrix: np.array, calibration matrix.
    :return: np.array, 2d human pose.
    """
    x_3d = pose_3d[:, 0]
    y_3d = pose_3d[:, 1]
    z_3d = pose_3d[:, 2]

    x_tof = (x_3d * calib_matrix[0, 0] + y_3d * calib_matrix[0, 1] + z_3d * calib_matrix[0, 2] + calib_matrix[0, 3]) / (
            x_3d * calib_matrix[2, 0] + y_3d * calib_matrix[2, 1] + z_3d * calib_matrix[2, 2] + calib_matrix[2, 3])
    y_tof = (x_3d * calib_matrix[1, 0] + y_3d * calib_matrix[1, 1] + z_3d * calib_matrix[1, 2] + calib_matrix[1, 3]) / (
            x_3d * calib_matrix[2, 0] + y_3d * calib_matrix[2, 1] + z_3d * calib_matrix[2, 2] + calib_matrix[2, 3])

    return np.array([x_tof, y_tof]).T


def build_dataset(fname, data_dir, num_subjects, num_scenes_per_subject, num_frames_per_scene, cam_id, calib_fname):
    """
    Build a Human3.6m subset.
    :param fname: string, resulting HDF5 file path
    :param data_dir: string, path in which Human3.5m data is stored
    :param num_subjects: int, number of subjects included in the subset built
    :param num_scenes_per_subject: int, number of scenes per subject included in the subset built
    :param num_frames_per_scene: int, number of consecutive frames per scene included in the subset built
    :param cam_id: string, 2d camera used ID
    :param calib_fname: string, calibration filename.
    :return: a HDF5 object
    """
    samples_loc = retrieve_samples_loc(data_dir, num_subjects, num_scenes_per_subject, cam_id)

    f = h5py.File(fname, mode="w")

    calib = np.load(calib_fname)

    f.attrs["data_dir"] = data_dir
    f.attrs["cam_id"] = cam_id
    f.attrs["calib"] = calib
    f.attrs["sorted_joints_idx"] = SORTED_JOINTS_IDX
    f.attrs["sorted_joints_names"] = SORTED_JOINTS_NAMES

    pose_dtype = np.dtype([
        ("subject", np.uint8, 1),
        ("scene", np.str_, 20),
        ("video", np.uint8, (num_frames_per_scene, 1000, 1000, 3)),
        ("tof", np.float32, (num_frames_per_scene, 144, 176)),
        ("pose_video", np.float32, (num_frames_per_scene, 14, 2)),
        ("pose_tof", np.float32, (num_frames_per_scene, 14, 2)),
        ("pose_3d", np.float32, (num_frames_per_scene, 14, 3)),
    ])
    data = f.create_dataset("data", (num_scenes_per_subject,), dtype=pose_dtype)

    for sample_idx, sample_loc in enumerate(samples_loc):
        subject = int(sample_loc["subject"])

        scene = sample_loc["scene"]
        print(scene)

        cap = cv2.VideoCapture(sample_loc["video"])
        nframes_total = int(cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT))
        max_init_frame = nframes_total - num_frames_per_scene
        init_frame = random.randint(0, max_init_frame)
        cap.set(cv2.cv2.CAP_PROP_POS_FRAMES, init_frame)
        video = np.array([cap.read()[1] for _ in range(num_frames_per_scene)])

        tof_raw = pycdf.CDF(sample_loc["tof"])
        tof_indices = np.array([int(tof_raw["Index"][0][init_frame + idx]) for idx in range(num_frames_per_scene)])
        tof = tof_raw["RangeFrames"][0][:, :, tof_indices].swapaxes(0, 2).swapaxes(1, 2)

        pose_video_raw = pycdf.CDF(sample_loc["pose_video"])
        pose_video = pose_video_raw["Pose"][0][init_frame:init_frame + num_frames_per_scene]
        pose_video = pose_video.reshape(num_frames_per_scene, 32, 2)
        pose_video = pose_video[:, SORTED_JOINTS_IDX]

        pose_3d_raw = pycdf.CDF(sample_loc["pose_3d"])
        pose_3d = pose_3d_raw["Pose"][0][init_frame:init_frame + num_frames_per_scene]
        pose_3d = pose_3d.reshape(num_frames_per_scene, 32, 3)
        pose_3d = pose_3d[:, SORTED_JOINTS_IDX]

        pose_tof = np.array([get_tof_coords(frame_pose, calib) for frame_pose in pose_3d])

        sample_data = np.array((subject, scene, video, tof, pose_video, pose_tof, pose_3d), dtype=pose_dtype)
        data[sample_idx] = sample_data

    return f


def play_sample(dataset, sample_idx):
    """
    Replay 2d video and TOF data, along with their registered poses for debugging purposes.
    :param dataset: HDF5 object, data.
    :param sample_idx: int, sample index.
    """
    sample = dataset["data"][sample_idx]
    fig = plt.figure()
    ax_video = fig.add_subplot(121)
    ax_tof = fig.add_subplot(122)
    for frame_idx in range(sample["video"].shape[0]):
        frame_video = sample["video"][frame_idx][:, :, ::-1]
        ax_video.imshow(frame_video)
        ax_video.axis('scaled')
        ax_video.set_xlim(0, frame_video.shape[1])
        ax_video.set_ylim(frame_video.shape[0], 0)
        ax_video.scatter(sample["pose_video"][frame_idx, :, 0], sample["pose_video"][frame_idx, :, 1])

        frame_tof = sample["tof"][frame_idx]
        ax_tof.imshow(frame_tof)
        ax_tof.set_xlim(0, frame_tof.shape[1])
        ax_tof.set_ylim(frame_tof.shape[0], 0)
        ax_tof.scatter(sample["pose_tof"][frame_idx, :, 0], sample["pose_tof"][frame_idx, :, 1])

        plt.pause(0.01)
        plt.draw()


if __name__ == "__main__":
    num_subjects = 1
    num_scenes_per_subject = 5
    num_frames_per_scene = 100
    cam_id = "55011271"
    data_dir = "data"
    calib_fname = os.path.join(data_dir, "calib.npy")
    out_fname = os.path.join(data_dir, "dataset.h5")

    # dataset = build_dataset(out_fname, data_dir, num_subjects, num_scenes_per_subject, num_frames_per_scene, cam_id,
    #                         calib_fname)
    # dataset.close()

    dataset = h5py.File(out_fname, mode="r")
    play_sample(dataset, sample_idx=2)

