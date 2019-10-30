import math
import os
import sys

sys.path.append("../../../../../")

import cv2
import h5py
from matplotlib import pyplot as plt
import numpy as np
from src.Estimator.Pose.pose_caffe import PoseEstimator
from src.Estimator.Human.human_naxvm import HumanDetector

LIMBS = np.array([1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13, 13, 14]).reshape((-1, 2)) - 1

COLORS = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0],
          [170, 255, 0], [255, 170, 0], [255, 0, 0], [255, 0, 170],
          [170, 0, 255]]


def draw_joints(im, bbox, joints, limbs=LIMBS, colors=COLORS, stickwidth=6):
    upper, lower = bbox
    cv2.rectangle(im, tuple(upper), tuple(lower), (0, 255, 0), 3)

    for i, (p, q) in enumerate(limbs):
        px, py = joints[p].astype(np.int)
        qx, qy = joints[q].astype(np.int)

        if px >= 0 and py >= 0 and qx >= 0 and qy >= 0:
            m_x = int(np.mean(np.array([px, qx])))
            m_y = int(np.mean(np.array([py, qy])))

            length = ((px - qx) ** 2. + (py - qy) ** 2.) ** 0.5
            angle = math.degrees(math.atan2(py - qy, px - qx))
            polygon = cv2.ellipse2Poly((m_x, m_y),
                                       (int(length / 2), stickwidth),
                                       int(angle), 0, 360, 1)
            cv2.fillConvexPoly(im, polygon, colors[i])

        if px >= 0 and py >= 0:
            cv2.circle(im, (px, py), 3, (0, 0, 0), -1)
        if qx >= 0 and qy >= 0:
            cv2.circle(im, (qx, qy), 3, (0, 0, 0), -1)

    return im


def estimate(frame, human, pose):
    bbox = human.get_bboxes(frame)[0]
    joints = pose.get_coords(frame, bbox)

    return bbox, joints


def read_hdf5(folder, subject_id, action_id, sequence_id):
    return h5py.File(os.path.join(folder, "cad120-%s_%s_%s.h5" % (subject_id, action_id, sequence_id)), mode="r")


def reorder_joints(joints):
    joints_cpm_order = np.zeros((16, 2), np.int)
    joints_cpm_order[13] = joints[14]
    joints_cpm_order[12] = joints[10]
    joints_cpm_order[11] = joints[9]
    joints_cpm_order[8] = joints[7]
    joints_cpm_order[9] = joints[8]
    joints_cpm_order[10] = joints[13]
    # joints_cpm_order[6] = joints[14]
    # joints_cpm_order[7] = joints[14]
    joints_cpm_order[1] = joints[1]
    joints_cpm_order[0] = joints[0]
    joints_cpm_order[7] = joints[12]
    joints_cpm_order[6] = joints[6]
    joints_cpm_order[5] = joints[5]
    joints_cpm_order[2] = joints[3]
    joints_cpm_order[3] = joints[4]
    joints_cpm_order[4] = joints[11]

    return joints_cpm_order


def get_dist(joint_pred, joint_real, hsize):
    x_pred, y_pred = joint_pred
    x_real, y_real = joint_real

    if x_real and y_real:
        dist = math.sqrt((x_pred - x_real) ** 2 + (y_pred - y_real) ** 2)
        dist /= hsize
    else:
        dist = np.nan

    return dist


def get_dists_2d(data, debug=True):
    rgb = data["rgb"]
    depth = data["depth"]
    anno = data["anno"]

    human_model = "../../../Human/models/naxvm/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb"
    pose_model = ["../../../Pose/models/caffe/pose_deploy_resize.prototxt",
                  "../../../Pose/models/caffe/pose_iter_320000.caffemodel"]
    boxsize = 192
    sigma = 21

    hd = HumanDetector(human_model, boxsize)
    pe = PoseEstimator(pose_model, boxsize, sigma, confidence_th=0.)

    dists = np.zeros(anno.shape)
    for i, (frame_rgb, frame_depth, frame_anno) in enumerate(zip(rgb, depth, anno)):
        joints_px_coords = reorder_joints(frame_anno["pixel_pos"])
        try:
            bbox_estimate, joints_estimate = estimate(frame_rgb.copy(), hd, pe)
            if debug:
                plt.figure()
                plt.subplot(121)
                plt.imshow(draw_joints(frame_rgb.copy(), ((0, 0), (640, 480)), joints_px_coords))
                plt.subplot(122)
                plt.imshow(draw_joints(frame_rgb.copy(), bbox_estimate, joints_estimate))
                plt.show()

            hsize = np.linalg.norm(joints_px_coords[1] - joints_px_coords[0])
            for j, (gt_joint, pred_joint) in enumerate(zip(joints_px_coords, joints_estimate)):
                dists[i, j] = get_dist(pred_joint, gt_joint, hsize)
        except:
            print("WARNING: No human detected (frame %03d)" % i)

    return dists


def evaluate_2d(data, debug=False):
    dists = get_dists_2d(data, debug)

    results = {"ankle": {"idx": [10, 13], "dists": [], "pct": 0, "curve": 0},
               "knee": {"idx": [9, 12], "dists": [], "pct": 0, "curve": 0},
               "hip": {"idx": [8, 11], "dists": [], "pct": 0, "curve": 0},
               "head": {"idx": [0, 1], "dists": [], "pct": 0, "curve": 0},
               "wrist": {"idx": [4, 7], "dists": [], "pct": 0, "curve": 0},
               "elbow": {"idx": [3, 6], "dists": [], "pct": 0, "curve": 0},
               "shoulder": {"idx": [2, 5], "dists": [], "pct": 0, "curve": 0},
               "total": {"idx": range(14), "dists": [], "pct": 0, "curve": 0}}

    max_th = 0.5
    step = 0.01
    thresholds = np.arange(0, max_th, step)

    print("\n-------- PCKh @ %.1f --------\n" % max_th)

    plt.figure()
    plt.suptitle("PCKh")
    for i, k in enumerate(results.keys()):
        idx = results[k]["idx"]

        k_dists = np.ravel(dists[:, idx])

        pck_joint = []
        for th in thresholds:
            n_correct = len([d for d in k_dists if d < th])
            n_total = len([d for d in k_dists if not np.isnan(d)])

            pck_joint.append(100 * float(n_correct) / n_total)

        results[k]["dists"] = k_dists
        results[k]["pct"] = pck_joint[-1]
        results[k]["curve"] = pck_joint

        print("\t%s: %.2f%%" % (k, pck_joint[-1]))

        plt.subplot(2, 4, i + 1)
        plt.plot(thresholds, pck_joint)

        plt.title("PCKh %s (%.2f%% @ %.1f)" % (k, pck_joint[-1], max_th))

        plt.xlim((0, max_th))
        plt.ylim((0, 100)), plt.ylabel("%")

    plt.show()


if __name__ == "__main__":
    subject_id = "1"
    action_id = "arranging_objects"
    sequence_id = "0510175411"

    global_folder = "/home/dpascualhe/repos/2017-tfm-david-pascual/src/Estimator/Tests/Datasets/CAD-120/"
    data = read_hdf5(global_folder, subject_id, action_id, sequence_id)
    evaluate_2d(data)
