import glob
import math
import os
import time

import cv2
import h5py
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import progressbar
from scipy.optimize import fsolve

from src.Estimator.kfilter import KFilter3D

HIPS_MAJOR_AX = 200
HIPS_MINOR_AX = 100

def get_pelvis(left_hip, right_hip, pelvis):
    x1, y1, z1 = left_hip
    x2, y2, z2 = right_hip
    x3, y3, z3 = pelvis
    def myFunction(z):
        x = z[0]
        y = z[1]
        F = np.empty((2))
        F[0] = (((x1 - x) ** 2) / (HIPS_MAJOR_AX ** 2)) + (((y1 - y) ** 2) / (HIPS_MINOR_AX ** 2)) - 1
        F[1] = (((x2 - x) ** 2) / (HIPS_MAJOR_AX ** 2)) + (((y2 - y) ** 2) / (HIPS_MINOR_AX ** 2)) - 1
        F[1] = (((x3 - x) ** 2) / (HIPS_MAJOR_AX ** 2)) + (((y3 - y) ** 2) / (HIPS_MINOR_AX ** 2)) - 1
        return F

    guess = np.array([0, 0])
    sol = fsolve(myFunction, guess)

    return np.array([sol[0], sol[1], (z1 + z2) / 2.])

def get_pelvis2(left_hip, right_hip, pelvis):
    bx = (left_hip[0] + right_hip[0]) / 2.
    by = (left_hip[1] + right_hip[1]) / 2.
    bz = (left_hip[2] + right_hip[2]) / 2.
    ax, ay, az = right_hip

    ab = np.linalg.norm(pelvis - right_hip)
    bc = (HIPS_MAJOR_AX + HIPS_MINOR_AX) / 4.
    ac = math.sqrt(ab**2 + bc**2)
    print(ab, bc, ac)

    cy = (ab**2 + ac**2 - bc**2) / (2 * ab)
    cx = math.sqrt(ac**2 - cy**2)

    return np.array([cx, cy, bz])

def get_depth_point(point, depth, calib_data=None, kfilter=None, search_area=11):
    """
    Get joints depth information.
    """
    height, width = depth.shape
    x, y = point

    depth = np.float32(depth)
    half_area = int(math.floor(search_area / 2.))
    roi = depth[y - half_area:y + half_area, x - half_area:x + half_area]

    while not np.count_nonzero(roi):
        half_area += 1
        roi = depth[y - half_area:y + half_area, x - half_area:x + half_area]

    roi[np.where(roi == 0)] = np.nan
    z = np.nanmin(roi)

    if x < half_area:
        x = half_area
    if x > width - half_area:
        x = width - half_area
    if y < half_area:
        y = half_area
    if y > height - half_area:
        y = height - half_area

    if calib_data:
        x = (x - calib_data["cx"]) * z / calib_data["fx"]
        y = (y - calib_data["cy"]) * z / calib_data["fy"]

    if kfilter is not None:
        kfilter.update()

    return x, y, z


def mpii_to_ours(joints, pelvis=None):
    new_joints = np.zeros((12, joints.shape[1]))

    new_joints[0] = np.mean(joints[:2], axis=0).astype(joints.dtype)  # head
    new_joints[1:7] = joints[2:8]
    new_joints[7:9] = joints[9:11]
    new_joints[9:11] = joints[12:14]
    new_joints[11] = np.mean(joints[np.array([8, 11])], axis=0).astype(joints.dtype)  # pelvis
    if joints.shape[1] == 2:
        new_joints[11] = np.mean(joints[np.array([8, 11])], axis=0).astype(joints.dtype)  # pelvis
    elif joints.shape[1] == 3:
        new_joints[11] = get_pelvis2(joints[8], joints[11], pelvis).astype(joints.dtype) # pelvis
        # new_joints[11] = np.mean(joints[np.array([8, 11])], axis=0).astype(joints.dtype) + 75  # pelvis
    else:
        raise Exception("ERROR: Invalid number of dimensions (%d)!" % joints.shape[1])

    return new_joints


def estimate(rgb, depth, config, calib, poses_gt=None, debug_video_fname="", search_area=11):
    """
    Estimate 3d coordinates given a set of data
    :param rgb: np.array, RGB video data
    :param depth: np.array, depth data
    :param calib: np.array, rgb video to depth video calibration
    :param poses_gt: np.array, labels in case human detection is going to be given by groundtruth
    :param debug_video_fname: str, filename to video with estimations
    :return: np.array, estimated human 3d poses
    """
    boxsize = config["boxsize"]
    human_framework = config["human_framework"]
    pose_framework = config["pose_framework"]

    available_human_fw = ["cpm_caffe", "cpm_tf", "naxvm"]
    available_pose_fw = ["cpm_caffe"]

    if human_framework == "label":
        if poses_gt is None:
            raise Exception("Label is needed for human detection!")
    else:
        human_model = config["human_models"][human_framework]

        if human_framework == "cpm_caffe":
            from src.Estimator.Human.human_cpm_caffe import HumanDetector
        elif human_framework == "cpm_tf":
            from src.Estimator.Human.human_cpm_tf import HumanDetector
        elif human_framework == "naxvm":
            from src.Estimator.Human.human_naxvm import HumanDetector
        else:
            print("'%s' is not supported for human detection" % human_framework)
            print("Available frameworks: " + str(available_human_fw))
            exit()

        hd = HumanDetector(human_model, boxsize, confidence_threshold=0.25)

    pose_model = config["pose_models"][pose_framework]
    if pose_framework == "cpm_caffe":
        from src.Estimator.Pose.pose_cpm import PoseCPM
        sigma = config["cpm_config"]["sigma"]
        pe = PoseCPM(pose_model, boxsize, sigma, confidence_th=0.25)
    elif pose_framework == "stacked":
        from src.Estimator.Pose.pose_stacked import PoseStacked
        pe = PoseStacked(pose_model, boxsize)
    elif pose_framework == "chained":
        from src.Estimator.Pose.pose_chained import PoseChained
        pe = PoseChained(pose_model, boxsize)
    else:
        print("'%s' is not supported for pose detection" % pose_framework)
        print("Available frameworks: " + str(available_pose_fw))
        exit()

    pos3d_kfilters = [KFilter3D() for _ in range(14)]

    num_frames = rgb.shape[0]

    poses_2d = np.empty((num_frames, 12, 2))
    poses_2d[:] = np.nan

    poses_3d = np.empty((num_frames, 12, 3))
    poses_3d[:] = np.nan

    bboxes = np.empty((num_frames, 2, 2), np.int32)
    bboxes[:] = -1

    bar = progressbar.ProgressBar()
    t_total = 0
    for frame_idx in bar(range(num_frames)):
        t_start = time.time()
        frame_rgb = rgb[frame_idx]
        frame_depth = depth[frame_idx]

        if human_framework == "label":
            pose_gt = np.squeeze(poses_gt)[frame_idx]
            cx, cy = np.mean(pose_gt, axis=0)
            max_x_diff = pose_gt[:, 0].max() - pose_gt[:, 0].min()
            max_y_diff = pose_gt[:, 1].max() - pose_gt[:, 1].min()
            bbox_radius = 1.25 * (np.max([max_x_diff, max_y_diff]) / 2)
            bbox = np.array([[cx - bbox_radius, cy - bbox_radius], [cx + bbox_radius, cy + bbox_radius]])
        else:
            bbox = hd.get_bboxes(frame_rgb)[:1][0]
        bboxes[frame_idx] = bbox.astype(np.int32)

        joints_2d = np.array(pe.get_coords(frame_rgb, bbox))
        if pose_framework == "cpm_caffe":
            joints_2d = joints_2d[:-1]  # bg joint

        # joints_2d = mpii_to_ours(joints_2d).astype(np.int)

        def get_joint_depth(idx):
            if np.count_nonzero(np.where(joints_2d[idx] == -1)):
                pos = (np.nan, np.nan, np.nan)
            else:
                pos = get_depth_point(joints_2d[idx], frame_depth, calib, search_area=search_area)

            return np.array(pos)

        def filter_joint_3d(idx, pos, mean_pos):
            pos = pos3d_kfilters[idx].update_filter(pos, mean_pos)
            return np.array(pos)

        joints_3d = []
        body_center = joints_2d[11]
        mean_pos = get_depth_point(body_center, frame_depth, calib, search_area=search_area)
        for idx in range(joints_2d.shape[0]):
            joint_3d = get_joint_depth(idx)
            # joints_3d.append(joint_3d)
            joints_3d.append(filter_joint_3d(idx, joint_3d, mean_pos))
        joints_3d = np.array(joints_3d)

        poses_2d[frame_idx] = mpii_to_ours(joints_2d).astype(np.int)

        joints_3d = mpii_to_ours(joints_3d, pelvis=mean_pos)
        poses_3d[frame_idx] = joints_3d
        t_total = t_total + (time.time() - t_start)
    print("Time! in ms: %f" % (t_total / num_frames))

    if debug_video_fname:
        fig = plt.figure()
        plt.suptitle("Predictions")
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection='3d')

        def animate(i):
            joints_2d, joints_3d, bbox = (poses_2d[i], poses_3d[i], bboxes[i])
            im = rgb[i]
            cv2.rectangle(im, tuple(bbox[0]), tuple(bbox[1]), (255, 0, 0))
            ax1.clear()
            ax1.scatter(joints_2d[:, 0], joints_2d[:, 1], s=50)
            for idx, joint_2d in enumerate(joints_2d):
                ax1.text(joint_2d[0] + 5, joint_2d[1] + 5, str(idx), color="red", fontsize=11)
            ax1.imshow(im)

            ax2.clear()
            ax2.scatter(joints_3d[:, 2], joints_3d[:, 0], -joints_3d[:, 1])
            ax2.set_xlim(2000, 4000)
            ax2.set_ylim(-1500, 500)
            ax2.set_zlim(-1000, 1000)
            ax2.view_init(30, i * 2)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani = animation.FuncAnimation(fig, animate, interval=1, save_count=num_frames)
        ani.save(debug_video_fname, writer=writer)

    return poses_2d, poses_3d


def store_estimation(pose, fname, data, dataset_fname):
    """
    Store 2D and 3D pose estimations
    :param poses: (np.array, np.array), 2D and 3D pose estimation
    :param fname: str, resulting filename
    :return: HDF5 object
    """
    f = h5py.File(fname, mode="w")

    poses_2d, poses_3d = pose
    num_poses = len(poses_2d)

    estimation_dtype = np.dtype([
        ("dataset_fname", np.str_, 200),
        ("camera_id", np.str_, 20),
        ("subject_id", np.str_, 20),
        ("action_id", np.str_, 20),
        ("repetition_id", np.str_, 20),
        ("frame_idx", np.uint16, (num_poses)),
        ("pose_2d", np.float32, (num_poses, 12, 2)),
        ("pose_3d", np.float32, (num_poses, 12, 3)),
    ])

    estimation = np.array((
        dataset_fname,
        data["camera_id"][0],
        data["subject_id"][0],
        data["action_id"][0],
        data["repetition_id"][0],
        np.uint16(range(num_poses)),
        np.float32(poses_2d),
        np.float32(poses_3d)
    ), dtype=estimation_dtype)

    f.create_dataset("estimation", (1,), estimation_dtype)
    f["estimation"][0] = estimation

    return f


if __name__ == "__main__":
    dataset_fname_key = "/home/dpascualhe/repos/2017-tfm-david-pascual/src/Estimator/Tests/Datasets/BMHAD/hdf5/*s02*.h5"
    outdir = "/home/dpascualhe/repos/2017-tfm-david-pascual/src/Estimator/Tests/Evaluation/preds/"
    human_framework = "label"
    pose_frameworks = ["cpm_caffe"]
    boxsize = 368

    test_name = "ours-cpm-fit_ellipse"

    for pose_framework in pose_frameworks:
        for dataset_fname in glob.glob(dataset_fname_key):
            print("\n\t%s" % dataset_fname)
            out_fname = os.path.join(outdir, "estimation-%s-%s-%s.h5" % (
                dataset_fname.split("_")[-1].split(".")[0], pose_framework, test_name))

            f = h5py.File(dataset_fname, mode="r")

            data = f["pose"]
            calibration = f["calibration"]
            cam_matrix = np.squeeze(calibration["K"])
            calibration_formated = {"fx": cam_matrix[0, 0], "fy": cam_matrix[1, 1],
                                    "cx": cam_matrix[0, 2], "cy": cam_matrix[1, 2]}

            config = {
                "human_framework": human_framework,
                "pose_framework": pose_framework,
                "human_models": {
                    "naxvm": "src/Estimator/Human/models/naxvm/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb"
                },
                "pose_models": {
                    "cpm_caffe": ["src/Estimator/Pose/models/caffe/pose_deploy_resize.prototxt",
                                  "src/Estimator/Pose/models/caffe/pose_iter_320000.caffemodel"],
                    "stacked": "src/Estimator/Pose/models/pytorch/simpleHG.pth",
                    "chained": "src/Estimator/Pose/models/pytorch/chained.pth"
                },
                "limbs": [1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13, 13, 14],
                "boxsize": boxsize,
                "cpm_config": {"sigma": 21}
            }

            debug_video_fname = out_fname.split(".")[0] + ".mp4"

            rgb_video = np.squeeze(data["rgb_video"])
            depth_video = np.squeeze(data["depth_video"])
            poses_estimated = estimate(rgb_video, depth_video, config, calibration_formated,
                                       data["pose_2d"],
                                       debug_video_fname="")
            estimations = store_estimation(poses_estimated, out_fname, data, dataset_fname)
            estimations.close()