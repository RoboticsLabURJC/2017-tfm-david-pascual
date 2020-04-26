from glob import glob
import os

import h5py
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd


def euclidian_distance(a, b):
    if np.count_nonzero(np.isnan([a, b])):
        return np.nan
    else:
        return np.linalg.norm(a - b)


def build_results_tables(
        pckh_2d_list, pckh_3d_list, mpjpe_list, pckh_2d_cam_act_list, pckh_3d_cam_act_list, mpjpe_cam_act_list,
        test_ids_pckh_2d=[], test_ids_pckh_3d=[], test_ids_mpjpe=[]
):
    print("TABLES:\n")

    # init tables
    df_joints_pckh_2d = pd.DataFrame({
        'method': test_ids_pckh_2d,
    })
    df_cam_act_pckh_2d = df_joints_pckh_2d.copy()
    df_joints_pckh_3d = pd.DataFrame({
        'method': test_ids_pckh_3d,
    })
    df_cam_act_pckh_3d = df_joints_pckh_3d.copy()
    df_joints_mpjpe = pd.DataFrame({
        'method': test_ids_mpjpe,
    })
    df_cam_act_mpjpe = df_joints_mpjpe.copy()

    # fill in table per joints pckh_2d
    if len(pckh_2d_list):
        for k in pckh_2d_list[0][0].keys():
            df_joints_pckh_2d[k] = [pckh_2d[0][k] for pckh_2d in pckh_2d_list]
        df_joints_pckh_2d["total"] = [pckh_2d[2] for pckh_2d in pckh_2d_list]
        print("PCKH 2D @ %.2f (%%) - Per joint" % pckh_3d_list[0][-1])
        print(df_joints_pckh_2d.to_string() + "\n")

    # fill in table per cam-action pair pckh_2d
    if len(pckh_2d_list):
        for k in pckh_2d_cam_act_list[0].keys():
            for j in pckh_2d_cam_act_list[0][k].keys():
                df_cam_act_pckh_2d[k + j] = [pckh_2d[k][j][0] for pckh_2d in pckh_2d_cam_act_list]
        df_cam_act_pckh_2d["total"] = [pckh_2d[2] for pckh_2d in pckh_2d_list]
        print("PCKH 2D @ %.2f (%%) - Cam/Action pair" % pckh_3d_list[0][-1])
        print(df_cam_act_pckh_2d.to_string() + "\n")
    
    # fill in table per joints pckh_3d
    if len(pckh_3d_list):
        for k in pckh_3d_list[0][0].keys():
            df_joints_pckh_3d[k] = [pckh_3d[0][k] for pckh_3d in pckh_3d_list]
        df_joints_pckh_3d["total"] = [pckh_3d[2] for pckh_3d in pckh_3d_list]
        print("PCKH 3D @ %.2f (%%) - Per joint" % pckh_3d_list[0][-1])
        print(df_joints_pckh_3d.to_string() + "\n")

    # fill in table per cam-action pair pckh_3d
    if len(pckh_3d_list):
        for k in pckh_3d_cam_act_list[0].keys():
            for j in pckh_3d_cam_act_list[0][k].keys():
                df_cam_act_pckh_3d[k + j] = [pckh_3d[k][j][0] for pckh_3d in pckh_3d_cam_act_list]
        df_cam_act_pckh_3d["total"] = [pckh_3d[2] for pckh_3d in pckh_3d_list]
        print("PCKH 3D @ %.2f (%%) - Cam/Action pair" % pckh_3d_list[0][-1])
        print(df_cam_act_pckh_3d.to_string() + "\n")
    
    # fill in table per joints mpjpe
    if len(mpjpe_list):
        for k in mpjpe_list[0][1].keys():
            df_joints_mpjpe[k] = [np.nanmean(mpjpe[1][k]) for mpjpe in mpjpe_list]
        df_joints_mpjpe["total"] = [mpjpe[0] for mpjpe in mpjpe_list]
        print("MPJPE (mm) - Per joint")
        print(df_joints_mpjpe.to_string() + "\n")

    # fill in table per cam-action pair mpjpe
    if len(mpjpe_list):
        for k in mpjpe_cam_act_list[0].keys():
            for j in mpjpe_cam_act_list[0][k].keys():
                df_cam_act_mpjpe[k + j] = [np.nanmean(mpjpe[k][j][0]) for mpjpe in mpjpe_cam_act_list]
        df_cam_act_mpjpe["total"] = [mpjpe[0] for mpjpe in mpjpe_list]
        print("MPJPE (mm) - Cam/Action pair")
        print(df_cam_act_mpjpe.to_string() + "\n")

    return 0



def get_mpjpe(preds, gt, joint_names):
    num_frames = gt.shape[0]

    pjpe = dict([(joint_name, np.nan) for joint_name in joint_names])
    pjpe_x = dict([(joint_name, np.nan) for joint_name in joint_names])
    pjpe_y = dict([(joint_name, np.nan) for joint_name in joint_names])
    pjpe_z = dict([(joint_name, np.nan) for joint_name in joint_names])

    pjpe_frame_by_frame = dict([(joint_name, []) for joint_name in joint_names])
    pjpe_frame_by_frame_x = dict([(joint_name, []) for joint_name in joint_names])
    pjpe_frame_by_frame_y = dict([(joint_name, []) for joint_name in joint_names])
    pjpe_frame_by_frame_z = dict([(joint_name, []) for joint_name in joint_names])

    for frame_idx in range(num_frames):
        skel_pred, skel_gt = (preds[frame_idx], gt[frame_idx])
        for idx, k in enumerate(joint_names):
            dist = euclidian_distance(skel_pred[idx], skel_gt[idx])
            pjpe_frame_by_frame[k].append(dist)
            pjpe_frame_by_frame_x[k].append(euclidian_distance(skel_pred[idx, 0], skel_gt[idx, 0]))
            pjpe_frame_by_frame_y[k].append(euclidian_distance(skel_pred[idx, 1], skel_gt[idx, 1]))
            pjpe_frame_by_frame_z[k].append(euclidian_distance(skel_pred[idx, 2], skel_gt[idx, 2]))
    pjpe_frame_by_frame_per_dim = (pjpe_frame_by_frame_x, pjpe_frame_by_frame_y, pjpe_frame_by_frame_z)

    for k in joint_names:
        pjpe[k] = np.mean(pjpe_frame_by_frame[k])
        pjpe_x[k] = np.mean(pjpe_frame_by_frame_x[k])
        pjpe_y[k] = np.mean(pjpe_frame_by_frame_y[k])
        pjpe_z[k] = np.mean(pjpe_frame_by_frame_z[k])
    pjpe_per_dim = (pjpe_x, pjpe_y, pjpe_z)

    mpjpe = np.mean(pjpe.values())
    mpjpe_x = np.mean(pjpe_x.values())
    mpjpe_y = np.mean(pjpe_y.values())
    mpjpe_z = np.mean(pjpe_z.values())
    mpjpe_per_dim = (mpjpe_x, mpjpe_y, mpjpe_z)

    return mpjpe, pjpe, pjpe_frame_by_frame, mpjpe_per_dim, pjpe_per_dim, pjpe_frame_by_frame_per_dim


def get_pckh(preds, gt, head_sizes, joint_names, max_th=0.5, step_th=0.01):
    num_frames = gt.shape[0]
    thresholds = np.arange(0, max_th + step_th, step_th)

    pckh = dict([(joint_name, dict([(th, np.nan) for th in thresholds])) for joint_name in joint_names])
    pckh_all_joints = dict([(th, np.nan) for th in thresholds])

    pckh_max_th = dict([(joint_name, np.nan) for joint_name in joint_names])

    norm_distances_frame_by_frame = dict([(joint_name, []) for joint_name in joint_names])
    norm_distances_frame_by_frame_all_joints = []

    for frame_idx in range(num_frames):
        skel_pred, skel_gt = (preds[frame_idx], gt[frame_idx])
        head_size = head_sizes[frame_idx]
        for idx, k in enumerate(joint_names):
            norm_distance = euclidian_distance(skel_pred[idx], skel_gt[idx]) / head_size
            norm_distances_frame_by_frame[k].append(norm_distance)
            norm_distances_frame_by_frame_all_joints.append(norm_distance)

    def get_pckh_single_joint(dists, th):
        correct = float(len([d for d in dists if (d < th and not np.isnan(d))]))
        total = float(len(dists))
        return 100 * (correct / total)

    for k in joint_names:
        for th in thresholds:
            pckh[k][th] = get_pckh_single_joint(norm_distances_frame_by_frame[k], th)
        if max_th not in thresholds:
            pckh_max_th[k] = get_pckh_single_joint(norm_distances_frame_by_frame[k], max_th)
        else:
            pckh_max_th[k] = pckh[k][max_th]

    for th in thresholds:
        pckh_all_joints[th] = get_pckh_single_joint(norm_distances_frame_by_frame_all_joints, th)
    if max_th not in thresholds:
        pckh_max_th_all_joints = get_pckh_single_joint(norm_distances_frame_by_frame_all_joints, max_th)
    else:
        pckh_max_th_all_joints = pckh_all_joints[max_th]

    return pckh_max_th_all_joints, pckh_all_joints, pckh_max_th, pckh


def store_scene_video(preds_2d, preds_3d, gt_2d, gt_3d, rgb, scene_id, test_id, outdir):
    fig = plt.figure()
    plt.suptitle("TEST: %s; SCENE: %s" % (test_id, scene_id))

    ax_gt_2d = fig.add_subplot(221)
    ax_gt_3d = fig.add_subplot(222, projection='3d')

    ax_pred_2d = fig.add_subplot(223)
    ax_pred_3d = fig.add_subplot(224, projection='3d')

    def animate(i):
        if preds_2d is not None:
            ax_gt_2d.clear()
            ax_gt_2d.set_title("Groundtruth - 2D")
            ax_gt_2d.scatter(gt_2d[i, :, 0], gt_2d[i, :, 1], s=50)
            for idx, joint_2d in enumerate(gt_2d[i]):
                ax_gt_2d.text(joint_2d[0] + 5, joint_2d[1] + 5, str(idx), color="red", fontsize=11)
            ax_gt_2d.imshow(rgb[i])

            ax_pred_2d.clear()
            ax_pred_2d.set_title("Prediction - 2D")
            ax_pred_2d.scatter(preds_2d[i, :, 0], preds_2d[i, :, 1], s=50)
            for idx, joint_2d in enumerate(preds_2d[i]):
                ax_pred_2d.text(joint_2d[0] + 5, joint_2d[1] + 5, str(idx), color="red", fontsize=11)
            ax_pred_2d.imshow(rgb[i])

        if preds_3d is not None:
            ax_gt_3d.clear()
            ax_gt_3d.set_title("Groundtruth - 3D")
            ax_gt_3d.scatter(gt_3d[i, :, 2], gt_3d[i, :, 0], -gt_3d[i, :, 1])
            ax_gt_3d.set_xlim(2000, 4000)
            ax_gt_3d.set_ylim(-1500, 500)
            ax_gt_3d.set_zlim(-1000, 1000)
            ax_gt_3d.view_init(30, i * 2)

            ax_pred_3d.clear()
            ax_pred_3d.set_title("Prediction - 3D")
            ax_pred_3d.scatter(preds_3d[i, :, 2], preds_3d[i, :, 0], -preds_3d[i, :, 1])
            ax_pred_3d.set_xlim(2000, 4000)
            ax_pred_3d.set_ylim(-1500, 500)
            ax_pred_3d.set_zlim(-1000, 1000)
            ax_pred_3d.view_init(30, i * 2)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani = animation.FuncAnimation(fig, animate, interval=1, save_count=rgb.shape[0])
    ani.save(os.path.join(outdir, "%s-%s.mp4" % (test_id, scene_id)), writer=writer)

def plot_pckh(pckh_max_th, pckh, pckh_max_th_all_joints, pckh_all_joints, scene_id, test_id, outdir, flag="2D"):
    fig = plt.figure()
    plt.suptitle("PCKh %s = %.2f%% - TEST: %s; SCENE: %s" % (flag, pckh_max_th_all_joints, test_id, scene_id), fontsize=21)

    for idx, k in enumerate(pckh.keys() + ["total"]):
        plt.subplot(3, 5, idx + 1)

        if k == "total":
            pts = np.array([(th, v) for th, v in sorted(zip(pckh_all_joints.keys(), pckh_all_joints.values()))])
            plt.plot(pts[:, 0], pts[:, 1], c="r")
            plt.title("%s - PCKh@%.1f = %.2f%%" % (k, pts[-1, 0], pckh_max_th_all_joints))
        else:
            pts = np.array([(th, v) for th, v in sorted(zip(pckh[k].keys(), pckh[k].values()))])
            plt.plot(pts[:, 0], pts[:, 1])
            plt.title("%s - PCKh@%.1f = %.2f%%" % (k, pts[-1, 0], pckh_max_th[k]))

        if idx > 5 * 2:
            plt.xlabel("threshold")
        if idx % 5 == 0:
            plt.ylabel("%")

        plt.ylim((0, 100))

    fig.set_size_inches(30, 15)
    plt.savefig(os.path.join(outdir, "pckh-%s-%s-%s" % (flag, test_id, scene_id)))


def plot_mpjpe(mpjpe, pjpe_frame_by_frame, mpjpe_per_dim, pjpe_frame_by_frame_per_dim, scene_id, test_id, outdir):
    for k in pjpe_frame_by_frame.keys():
        pjpe_frame_by_frame[k] = np.array(pjpe_frame_by_frame[k])[~np.isnan(pjpe_frame_by_frame[k])]
    
    for pjpe_dim in pjpe_frame_by_frame_per_dim:
        for k in pjpe_dim.keys():
            pjpe_dim[k] = np.array(pjpe_dim[k])[~np.isnan(pjpe_dim[k])]

    fig = plt.figure()

    plt.suptitle("MPJPE = %.2f mm - TEST: %s; SCENE: %s" % (mpjpe, test_id, scene_id), fontsize=21)

    ax = plt.subplot(211)
    ax.boxplot(pjpe_frame_by_frame.values())
    ax.axhline(mpjpe, c="green", linewidth=2, linestyle="--")
    ax.set_xticklabels(pjpe_frame_by_frame.keys(), rotation=45, fontsize=17)

    ax_x = plt.subplot(234)
    ax_x.set_title("MPJPE = %2.f mm - X dim" % mpjpe_per_dim[0])
    ax_x.boxplot(pjpe_frame_by_frame_per_dim[0].values())
    ax_x.axhline(mpjpe_per_dim[0], c="green", linewidth=2, linestyle="--")
    ax_x.set_xticklabels(pjpe_frame_by_frame_per_dim[0].keys(), rotation=45, fontsize=17)
    
    ax_y = plt.subplot(235)
    ax_y.set_title("MPJPE = %2.f mm - Y dim" % mpjpe_per_dim[1])
    ax_y.boxplot(pjpe_frame_by_frame_per_dim[1].values())
    ax_y.axhline(mpjpe_per_dim[1], c="green", linewidth=2, linestyle="--")
    ax_y.set_xticklabels(pjpe_frame_by_frame_per_dim[1].keys(), rotation=45, fontsize=17)
    
    ax_z = plt.subplot(236)
    ax_z.set_title("MPJPE = %2.f mm - Z dim" % mpjpe_per_dim[2])
    ax_z.boxplot(pjpe_frame_by_frame_per_dim[2].values())
    ax_z.axhline(mpjpe_per_dim[2], c="green", linewidth=2, linestyle="--")
    ax_z.set_xticklabels(pjpe_frame_by_frame_per_dim[2].keys(), rotation=45, fontsize=17)

    fig.set_size_inches(30, 25)
    plt.savefig(os.path.join(outdir, "mpjpe-%s-%s" % (test_id, scene_id)))

def plot_multiple_pckh(pckh_max_th_list, pckh_list, pckh_max_th_all_joints_list, pckh_all_joints_list, test_id_list, outdir, flag="2D"):
    all_data = (pckh_max_th_list, pckh_list, pckh_max_th_all_joints_list, pckh_all_joints_list, test_id_list)

    fig = plt.figure()
    plt.suptitle("PCKh %s" % flag, fontsize=21)

    for j in range(len(test_ids)):
        pckh_max_th = all_data[0][j]
        pckh = all_data[1][j]
        pckh_max_th_all_joints = all_data[2][j]
        pckh_all_joints = all_data[3][j]
        test_id = all_data[4][j]

        for idx, k in enumerate(pckh.keys() + ["total"]):
            if k == "total":
                plt.subplot(3, 5, 15)
                pts = np.array([(th, v) for th, v in sorted(zip(pckh_all_joints.keys(), pckh_all_joints.values()))])
            else:
                plt.subplot(3, 5, idx + 1)
                pts = np.array([(th, v) for th, v in sorted(zip(pckh[k].keys(), pckh[k].values()))])
            plt.plot(pts[:, 0], pts[:, 1], label=test_id)
            plt.title("%s" % k)

            if idx > 5 * 2:
                plt.xlabel("threshold")
            if idx % 5 == 0:
                plt.ylabel("%")

            plt.ylim((0, 100))
        plt.legend()
    fig.set_size_inches(30, 15)
    plt.savefig(os.path.join(outdir, "pckh-%s-all" % (flag)))

def evaluate(preds_fnames, test_id, outdir, metrics):
    metrics_available = ["video", "pckh_2d", "pckh_3d", "mpjpe"]
    for metric in metrics:
        if metric not in metrics_available:
            print("ERROR: Unknown metric '%s'\n\tMust be one of %s" % (metric, str(metrics_available)))

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    outdir = os.path.join(outdir, "individual_scenes")
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    pred_poses_2d = None
    pred_poses_3d = None
    unmasked_pred_poses_3d = None
    gt_poses_2d = None
    gt_poses_3d = None
    unmasked_gt_poses_3d = None
    head_sizes_2d = None
    head_sizes_3d = None

    scenes_evaluated = []
    for pred_fname in preds_fnames:
        # Open HDF5 files
        pred = h5py.File(pred_fname, mode="r")["estimation"]
        gt_fname = pred["dataset_fname"][0]
        gt_aux = h5py.File(gt_fname, mode="r")
        gt = gt_aux["pose"]

        # Get scene id
        scene_id = pred["subject_id"][0] + pred["action_id"][0] + pred["repetition_id"][0] + pred["camera_id"][0]
        if scene_id in scenes_evaluated:
            print("WARNING: Ignoring %s\n\tMultiple predictions for the same scene not allowed!" % pred_fname)
            continue

        # Get masked poses array
        masked_poses_fname = os.path.join(os.path.dirname(pred_fname), "masked_poses-%s*.npy" % pred["subject_id"][0])
        masked_poses = np.load(glob(masked_poses_fname)[0])

        # Create scene dir
        scene_dir = os.path.join(outdir, scene_id)
        if not os.path.isdir(scene_dir):
            os.mkdir(scene_dir)
        scenes_evaluated.append(scene_id)

        # Get 2D poses
        if "pckh_2d" in metrics and "pose_2d" in pred.dtype.names:
            print(scene_dir)
            pred_poses_2d = np.squeeze(pred["pose_2d"])
            gt_poses_2d = np.squeeze(gt["pose_2d"])
            head_sizes_2d = np.squeeze(gt["head_sizes_2d"])

        # Get 3D poses
        if "pckh_3d" in metrics or "mpjpe" in metrics and "pose_3d" in pred.dtype.names:
            unmasked_pred_poses_3d = np.squeeze(pred["pose_3d"])
            pred_poses_3d = unmasked_pred_poses_3d[masked_poses]
            unmasked_gt_poses_3d = np.squeeze(gt["pose_3d"])
            gt_poses_3d = unmasked_gt_poses_3d[masked_poses]
            head_sizes_3d = np.squeeze(gt["head_sizes_3d"])

        # Get joint names
        jnames = list(np.squeeze(gt_aux.attrs["joint_order"]))

        print("-- Scene %s --" % scene_id)

        # Store videos for debugging
        if "video" in metrics:
            rgb_video = np.squeeze(gt["rgb_video"])
            store_scene_video(
                pred_poses_2d, unmasked_pred_poses_3d, gt_poses_2d, unmasked_gt_poses_3d, rgb_video, scene_id, test_id, scene_dir
            )

        # PCKh 2D
        max_th_pckh_2d = 1.
        if "pckh_2d" in metrics and pred_poses_2d is not None:
            pckh_max_th_all_joints_2d, pckh_all_joints_2d, pckh_max_th_2d, pckh_2d = get_pckh(pred_poses_2d,
                                                                                              gt_poses_2d,
                                                                                              head_sizes_2d,
                                                                                              jnames,
                                                                                              max_th=max_th_pckh_2d)
            plot_pckh(pckh_max_th_2d, pckh_2d, pckh_max_th_all_joints_2d, pckh_all_joints_2d, scene_id, test_id,
                      scene_dir,
                      flag="2D")
            print("\tPCKh 2D @ %.1f = %.2f%%" % (max_th_pckh_2d, pckh_max_th_all_joints_2d))

        # PCKh 3D
        max_th_pckh_3d = 1.
        if "pckh_3d" in metrics and unmasked_pred_poses_3d is not None:
            pckh_max_th_all_joints_3d, pckh_all_joints_3d, pckh_max_th_3d, pckh_3d = get_pckh(unmasked_pred_poses_3d,
                                                                                              unmasked_gt_poses_3d,
                                                                                              head_sizes_3d,
                                                                                              jnames,
                                                                                              max_th=max_th_pckh_3d)
            plot_pckh(pckh_max_th_3d, pckh_3d, pckh_max_th_all_joints_3d, pckh_all_joints_3d, scene_id, test_id,
                      scene_dir,
                      flag="3D")
            print("\tPCKh 3D @ %.1f = %.2f%%" % (max_th_pckh_3d, pckh_max_th_all_joints_3d))

        # MPJPE
        if "mpjpe" in metrics and pred_poses_3d is not None:
            mpjpe, pjpe, pjpe_frame_by_frame, mpjpe_per_dim, pjpe_per_dim, pjpe_frame_by_frame_per_dim = get_mpjpe(
                pred_poses_3d, gt_poses_3d, jnames
            )
            plot_mpjpe(
                mpjpe, pjpe_frame_by_frame, mpjpe_per_dim, pjpe_frame_by_frame_per_dim, scene_id, test_id, scene_dir
            )
            print("\tMPJPE = %.2fmm\n\t\tx = %.2f mm\n\t\ty = %.2f mm\n\t\tz = %.2f mm" %
                  (mpjpe, mpjpe_per_dim[0], mpjpe_per_dim[1], mpjpe_per_dim[2]))


def evaluate_all_preds(preds_fnames, test_id, outdir, metrics):
    metrics_available = ["pckh_2d", "pckh_3d", "mpjpe"]
    for metric in metrics:
        if metric not in metrics_available:
            print("WARNING: Unknown metric '%s'\n\tMust be one of %s" % (metric, str(metrics_available)))

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    pred_poses_2d = None
    pred_poses_3d = None
    unmasked_pred_poses_3d = None
    gt_poses_2d = None
    gt_poses_3d = None
    unmasked_gt_poses_3d = None
    head_sizes_2d = None
    head_sizes_3d = None
    jnames = None

    # Set scene id to all_scenes
    scene_id = "all_scenes"

    # Create all scenes dir
    scene_dir = os.path.join(outdir, scene_id)
    if not os.path.isdir(scene_dir):
        os.mkdir(scene_dir)

    scenes_evaluated = []

    for pred_fname in preds_fnames:
        # Open HDF5 files
        pred = h5py.File(pred_fname, mode="r")["estimation"]
        gt_fname = pred["dataset_fname"][0]
        gt_aux = h5py.File(gt_fname, mode="r")
        gt = gt_aux["pose"]

        # Get scene id
        current_scene_id = pred["subject_id"][0] + pred["action_id"][0] + pred["repetition_id"][0] + pred["camera_id"][0]
        if scene_id in scenes_evaluated:
            print("WARNING: Ignoring %s\n\tMultiple predictions for the same scene not allowed!" % pred_fname)
            continue
        scenes_evaluated.append(current_scene_id)

        # Get masked poses array
        masked_poses_fname = os.path.join(os.path.dirname(pred_fname), "masked_poses-%s*.npy" % pred["subject_id"][0])
        masked_poses = np.load(glob(masked_poses_fname)[0])

        # Get 2D poses
        if "pckh_2d" in metrics and "pose_2d" in pred.dtype.names:
            if pred_poses_2d is None:
                pred_poses_2d = np.squeeze(pred["pose_2d"])
            else:
                pred_poses_2d = np.append(pred_poses_2d, np.squeeze(pred["pose_2d"]), axis=0)

            if gt_poses_2d is None:
                gt_poses_2d = np.squeeze(gt["pose_2d"])
            else:
                gt_poses_2d = np.append(gt_poses_2d, np.squeeze(gt["pose_2d"]), axis=0)

            if head_sizes_2d is None:
                head_sizes_2d = np.squeeze(gt["head_sizes_2d"])
            else:
                head_sizes_2d = np.append(head_sizes_2d, np.squeeze(gt["head_sizes_2d"]), axis=0)

        # Get 3D poses
        if "pckh_3d" in metrics or "mpjpe" in metrics and "pose_3d" in pred.dtype.names:
            if pred_poses_3d is None:
                unmasked_pred_poses_3d = np.squeeze(pred["pose_3d"])
                pred_poses_3d = unmasked_pred_poses_3d[masked_poses]
            else:
                unmasked_pred_poses_3d = np.append(unmasked_pred_poses_3d, np.squeeze(pred["pose_3d"]), axis=0)
                pred_poses_3d = np.append(pred_poses_3d, np.squeeze(pred["pose_3d"])[masked_poses], axis=0)

            if gt_poses_3d is None:
                unmasked_gt_poses_3d = np.squeeze(gt["pose_3d"])
                gt_poses_3d = unmasked_gt_poses_3d[masked_poses]
            else:
                unmasked_gt_poses_3d = np.append(unmasked_gt_poses_3d, np.squeeze(gt["pose_3d"]), axis=0)
                gt_poses_3d = np.append(gt_poses_3d, np.squeeze(gt["pose_3d"])[masked_poses], axis=0)

            if head_sizes_3d is None:
                head_sizes_3d = np.squeeze(gt["head_sizes_3d"])
            else:
                head_sizes_3d = np.append(head_sizes_3d, np.squeeze(gt["head_sizes_3d"]), axis=0)

        # Get joint names
        if jnames is None:
            jnames = list(np.squeeze(gt_aux.attrs["joint_order"]))
        else:
            if jnames != list(np.squeeze(gt_aux.attrs["joint_order"])):
                raise Exception("ERROR: Mismatch in joints to be evaluated!")

    print("-- All scenes --")

    # PCKh 2D
    max_th_pckh_2d = 1.
    pckh_2d_tuple = ()
    if "pckh_2d" in metrics and pred_poses_2d is not None:
        pckh_max_th_all_joints_2d, pckh_all_joints_2d, pckh_max_th_2d, pckh_2d = get_pckh(pred_poses_2d,
                                                                                          gt_poses_2d,
                                                                                          head_sizes_2d,
                                                                                          jnames,
                                                                                          max_th=max_th_pckh_2d)
        plot_pckh(pckh_max_th_2d, pckh_2d, pckh_max_th_all_joints_2d, pckh_all_joints_2d, scene_id, test_id,
                  scene_dir,
                  flag="2D")
        print("\tPCKh 2D @ %.1f = %.2f%%" % (max_th_pckh_2d, pckh_max_th_all_joints_2d))

        pckh_2d_tuple = (pckh_max_th_2d, pckh_2d, pckh_max_th_all_joints_2d, pckh_all_joints_2d, max_th_pckh_2d)

    # PCKh 3D
    max_th_pckh_3d = 1.
    pckh_3d_tuple = ()
    if "pckh_3d" in metrics and unmasked_pred_poses_3d is not None:
        pckh_max_th_all_joints_3d, pckh_all_joints_3d, pckh_max_th_3d, pckh_3d = get_pckh(unmasked_pred_poses_3d,
                                                                                          unmasked_gt_poses_3d,
                                                                                          head_sizes_3d,
                                                                                          jnames,
                                                                                          max_th=max_th_pckh_3d)
        plot_pckh(pckh_max_th_3d, pckh_3d, pckh_max_th_all_joints_3d, pckh_all_joints_3d, scene_id, test_id,
                  scene_dir,
                  flag="3D")
        print("\tPCKh 3D @ %.1f = %.2f%%" % (max_th_pckh_3d, pckh_max_th_all_joints_3d))

        pckh_3d_tuple = (pckh_max_th_3d, pckh_3d, pckh_max_th_all_joints_3d, pckh_all_joints_3d, max_th_pckh_3d)

    # MPJPE
    mpjpe_tuple = ()
    if "mpjpe" in metrics and pred_poses_3d is not None:
        mpjpe, pjpe, pjpe_frame_by_frame, mpjpe_per_dim, pjpe_per_dim, pjpe_frame_by_frame_per_dim = get_mpjpe(
            pred_poses_3d, gt_poses_3d, jnames
        )
        plot_mpjpe(
            mpjpe, pjpe_frame_by_frame, mpjpe_per_dim, pjpe_frame_by_frame_per_dim, scene_id, test_id, scene_dir
        )
        print("\tMPJPE = %.2fmm\n\t\tx = %.2f mm\n\t\ty = %.2f mm\n\t\tz = %.2f mm" %
              (mpjpe, mpjpe_per_dim[0], mpjpe_per_dim[1], mpjpe_per_dim[2]))

        mpjpe_tuple = (mpjpe, pjpe_frame_by_frame, mpjpe_per_dim, pjpe_frame_by_frame_per_dim)

    return pckh_2d_tuple, pckh_3d_tuple, mpjpe_tuple



def evaluate_per_action_camera_pair(preds_fnames, test_id, outdir, metrics):
    metrics_available = ["pckh_2d", "pckh_3d", "mpjpe"]
    for metric in metrics:
        if metric not in metrics_available:
            print("WARNING: Unknown metric '%s'\n\tMust be one of %s" % (metric, str(metrics_available)))

    outdir = os.path.join(outdir, "cam_action_pairs")
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    pred_poses_2d = {}
    pred_poses_3d = {}
    unmasked_pred_poses_3d = {}
    gt_poses_2d = {}
    gt_poses_3d = {}
    unmasked_gt_poses_3d = {}
    head_sizes_2d = {}
    head_sizes_3d = {}
    jnames = None

    for pred_fname in preds_fnames:
        # Open HDF5 files
        pred = h5py.File(pred_fname, mode="r")["estimation"]
        gt_fname = pred["dataset_fname"][0]
        gt_aux = h5py.File(gt_fname, mode="r")
        gt = gt_aux["pose"]

        camera_id = pred["camera_id"][0]
        action_id = pred["action_id"][0]

        if jnames is None:
            jnames = list(np.squeeze(gt_aux.attrs["joint_order"]))
        else:
            if jnames != list(np.squeeze(gt_aux.attrs["joint_order"])):
                raise Exception("ERROR: Mismatch in joints to be evaluated!")

        # Get masked poses array
        masked_poses_fname = os.path.join(os.path.dirname(pred_fname), "masked_poses-%s*.npy" % pred["subject_id"][0])
        masked_poses = np.load(glob(masked_poses_fname)[0])

        # Get 2d poses
        if "pckh_2d" in metrics and "pose_2d" in pred.dtype.names:
            if camera_id not in pred_poses_2d.keys():
                pred_poses_2d[camera_id] = {}
            if camera_id not in gt_poses_2d.keys():
                gt_poses_2d[camera_id] = {}
            if camera_id not in head_sizes_2d.keys():
                head_sizes_2d[camera_id] = {}

            if action_id not in pred_poses_2d[camera_id].keys():
                pred_poses_2d[camera_id][action_id] = None
            if action_id not in gt_poses_2d[camera_id].keys():
                gt_poses_2d[camera_id][action_id] = None
            if action_id not in head_sizes_2d[camera_id].keys():
                head_sizes_2d[camera_id][action_id] = None

            if pred_poses_2d[camera_id][action_id] is None:
                pred_poses_2d[camera_id][action_id] = np.squeeze(pred["pose_2d"])
            else:
                pred_poses_2d[camera_id][action_id] = np.append(pred_poses_2d[camera_id][action_id], np.squeeze(pred["pose_2d"]))
            if gt_poses_2d[camera_id][action_id] is None:
                gt_poses_2d[camera_id][action_id] = np.squeeze(gt["pose_2d"])
            else:
                gt_poses_2d[camera_id][action_id] = np.append(gt_poses_2d[camera_id][action_id], np.squeeze(gt["pose_2d"]))
            if head_sizes_2d[camera_id][action_id] is None:
                head_sizes_2d[camera_id][action_id] = np.squeeze(gt["head_sizes_2d"])
            else:
                head_sizes_2d[camera_id][action_id] = np.append(head_sizes_2d[camera_id][action_id], np.squeeze(gt["head_sizes_2d"]))


        # Get 3d poses
        if "pckh_3d" in metrics or "mpjpe" in metrics and "pose_3d" in pred.dtype.names:
            if camera_id not in pred_poses_3d.keys():
                pred_poses_3d[camera_id] = {}
            if camera_id not in unmasked_pred_poses_3d.keys():
                unmasked_pred_poses_3d[camera_id] = {}
            if camera_id not in gt_poses_3d.keys():
                gt_poses_3d[camera_id] = {}
            if camera_id not in unmasked_gt_poses_3d.keys():
                unmasked_gt_poses_3d[camera_id] = {}
            if camera_id not in head_sizes_3d.keys():
                head_sizes_3d[camera_id] = {}

            if action_id not in pred_poses_3d[camera_id].keys():
                pred_poses_3d[camera_id][action_id] = None
            if action_id not in unmasked_pred_poses_3d[camera_id].keys():
                unmasked_pred_poses_3d[camera_id][action_id] = None
            if action_id not in gt_poses_3d[camera_id].keys():
                gt_poses_3d[camera_id][action_id] = None
            if action_id not in unmasked_gt_poses_3d[camera_id].keys():
                unmasked_gt_poses_3d[camera_id][action_id] = None
            if action_id not in head_sizes_3d[camera_id].keys():
                head_sizes_3d[camera_id][action_id] = None

            if pred_poses_3d[camera_id][action_id] is None:
                unmasked_pred_poses_3d[camera_id][action_id] = np.squeeze(pred["pose_3d"])
                pred_poses_3d[camera_id][action_id] = unmasked_pred_poses_3d[camera_id][action_id][masked_poses]
            else:
                unmasked_pred_poses_3d[camera_id][action_id] = np.append(unmasked_pred_poses_3d[camera_id][action_id], np.squeeze(pred["pose_3d"]), axis=0)
                pred_poses_3d[camera_id][action_id] = np.append(pred_poses_3d[camera_id][action_id], np.squeeze(pred["pose_3d"])[masked_poses], axis=0)
            if gt_poses_3d[camera_id][action_id] is None:
                unmasked_gt_poses_3d[camera_id][action_id] = np.squeeze(gt["pose_3d"])
                gt_poses_3d[camera_id][action_id] = unmasked_gt_poses_3d[camera_id][action_id][masked_poses]
            else:
                unmasked_gt_poses_3d[camera_id][action_id] = np.append(unmasked_gt_poses_3d[camera_id][action_id], np.squeeze(gt["pose_3d"]), axis=0)
                gt_poses_3d[camera_id][action_id] = np.append(gt_poses_3d[camera_id][action_id], np.squeeze(gt["pose_3d"])[masked_poses], axis=0)
            if head_sizes_3d[camera_id][action_id] is None:
                head_sizes_3d[camera_id][action_id] = np.squeeze(gt["head_sizes_3d"])
            else:
                head_sizes_3d[camera_id][action_id] = np.append(head_sizes_3d[camera_id][action_id], np.squeeze(gt["head_sizes_3d"]))

    print("-- Camera/action pair --")

    # 2D
    max_th_pckh_2d = 1.
    pckh_2d_cam_act = {}
    if pred_poses_2d is not None:
        for camera_id in pred_poses_2d.keys():
            for action_id in pred_poses_2d[camera_id].keys():
                cam_act_id = camera_id + action_id
                cam_act_dir = os.path.join(outdir, cam_act_id)
                if not os.path.isdir(cam_act_dir):
                    os.mkdir(cam_act_dir)

                pred_poses_2d_cam_act = pred_poses_2d[camera_id][action_id]
                gt_poses_2d_cam_act = gt_poses_2d[camera_id][action_id]
                head_sizes_2d_cam_act = head_sizes_2d[camera_id][action_id]

                # PCKh 2D
                if "pckh_2d" in metrics:
                    pckh_max_th_all_joints_2d, pckh_all_joints_2d, pckh_max_th_2d, pckh_2d = get_pckh(pred_poses_2d_cam_act,
                                                                                                      gt_poses_2d_cam_act,
                                                                                                      head_sizes_2d_cam_act,
                                                                                                      jnames,
                                                                                                      max_th=max_th_pckh_2d)

                    plot_pckh(pckh_max_th_2d, pckh_2d, pckh_max_th_all_joints_2d, pckh_all_joints_2d, cam_act_id, test_id,
                              cam_act_dir,
                              flag="2D")

                    if camera_id not in pckh_2d_cam_act.keys():
                        pckh_2d_cam_act[camera_id] = {}
                    pckh_2d_cam_act[camera_id][action_id] = \
                        pckh_max_th_all_joints_2d, pckh_all_joints_2d, pckh_max_th_2d, pckh_2d, max_th_pckh_2d
                    print("\tPCKh 2D %s, %s @ %.1f = %.2f%%" %
                          (camera_id, action_id, max_th_pckh_2d, pckh_max_th_all_joints_2d))

    max_th_pckh_3d = 1.
    pckh_3d_cam_act = {}
    mpjpe_cam_act = {}
    if pred_poses_3d is not None:
        for camera_id in pred_poses_3d.keys():
            for action_id in pred_poses_3d[camera_id].keys():
                cam_act_id = camera_id + action_id
                cam_act_dir = os.path.join(outdir, cam_act_id)
                if not os.path.isdir(cam_act_dir):
                    os.mkdir(cam_act_dir)

                pred_poses_3d_cam_act = pred_poses_3d[camera_id][action_id]
                unmasked_pred_poses_3d_cam_act = unmasked_pred_poses_3d[camera_id][action_id]
                gt_poses_3d_cam_act = gt_poses_3d[camera_id][action_id]
                unmasked_gt_poses_3d_cam_act = unmasked_gt_poses_3d[camera_id][action_id]
                head_sizes_3d_cam_act = head_sizes_3d[camera_id][action_id]

                # PCKh 3D
                if "pckh_3d" in metrics:
                    pckh_max_th_all_joints_3d, pckh_all_joints_3d, pckh_max_th_3d, pckh_3d = get_pckh(unmasked_pred_poses_3d_cam_act,
                                                                                                      unmasked_gt_poses_3d_cam_act,
                                                                                                      head_sizes_3d_cam_act,
                                                                                                      jnames,
                                                                                                      max_th=max_th_pckh_3d)

                    plot_pckh(pckh_max_th_3d, pckh_3d, pckh_max_th_all_joints_3d, pckh_all_joints_3d, cam_act_id, test_id,
                              cam_act_dir,
                              flag="3D")

                    if camera_id not in pckh_3d_cam_act.keys():
                        pckh_3d_cam_act[camera_id] = {}
                    pckh_3d_cam_act[camera_id][action_id] = \
                        pckh_max_th_all_joints_3d, pckh_all_joints_3d, pckh_max_th_3d, pckh_3d, max_th_pckh_3d
                    print("\tPCKh 3D %s, %s @ %.1f = %.2f%%" %
                          (camera_id, action_id, max_th_pckh_3d, pckh_max_th_all_joints_3d))

                # MPJPE
                if "mpjpe" in metrics:
                    mpjpe, pjpe, pjpe_frame_by_frame, mpjpe_per_dim, pjpe_per_dim, pjpe_frame_by_frame_per_dim = get_mpjpe(
                        pred_poses_3d_cam_act, gt_poses_3d_cam_act, jnames
                    )

                    plot_mpjpe(
                        mpjpe, pjpe_frame_by_frame, mpjpe_per_dim, pjpe_frame_by_frame_per_dim, cam_act_id, test_id,
                        cam_act_dir
                    )

                    if camera_id not in mpjpe_cam_act.keys():
                        mpjpe_cam_act[camera_id] = {}
                    mpjpe_cam_act[camera_id][action_id] = \
                        mpjpe, pjpe, pjpe_frame_by_frame, mpjpe_per_dim, pjpe_per_dim, pjpe_frame_by_frame_per_dim
                    print("\tMPJPE %s, %s = %.2fmm\n\t\tx = %.2f mm\n\t\ty = %.2f mm\n\t\tz = %.2f mm" %
                          (camera_id, action_id, mpjpe, mpjpe_per_dim[0], mpjpe_per_dim[1], mpjpe_per_dim[2]))

    return pckh_2d_cam_act, pckh_3d_cam_act, mpjpe_cam_act


if __name__ == "__main__":
    # test_ids = ["ours-cpm-new_skel", "ours-stacked_hourglass", "ours-chained_predictions", "zimmermann-new_skel", "zimmermann-kfilter"]
    test_ids = ["ours-cpm-fit_ellipse"]
    global_dir = "/home/dpascualhe/repos/2017-tfm-david-pascual/src/Estimator/Tests/Evaluation/results/dummy"
    if not os.path.isdir(global_dir):
        os.mkdir(global_dir)
    metrics = ["pckh_2d", "pckh_3d", "mpjpe"]

    pckh_2d_list = []
    pckh_3d_list = []
    mpjpe_list = []
    pckh_2d_cam_act_list = []
    pckh_3d_cam_act_list = []
    mpjpe_cam_act_list = []
    test_ids_pckh_2d, test_ids_pckh_3d, test_ids_mpjpe = ([], [], [])
    for test_id in test_ids:
        preds_fname_key = "/home/dpascualhe/repos/2017-tfm-david-pascual/src/Estimator/Tests/Evaluation/preds/*s02*%s.h5" % test_id
        outdir = os.path.join(global_dir, test_id)

        print("---------------\n%s\n---------------" % test_id)
        evaluate(list(glob(preds_fname_key)), test_id, outdir, metrics)
        pckh_2d, pckh_3d, mpjpe = evaluate_all_preds(list(glob(preds_fname_key)), test_id, outdir, metrics)
        if len(pckh_2d):
            pckh_2d_list.append(pckh_2d)
            test_ids_pckh_2d.append(test_id)
        if len(pckh_3d):
            pckh_3d_list.append(pckh_3d)
            test_ids_pckh_3d.append(test_id)
        if len(mpjpe):
            mpjpe_list.append(mpjpe)
            test_ids_mpjpe.append(test_id)

        pckh_2d_cam_act, pckh_3d_cam_act, mpjpe_cam_act =\
            evaluate_per_action_camera_pair(list(glob(preds_fname_key)), test_id, outdir, metrics)
        if len(pckh_2d_cam_act):
            pckh_2d_cam_act_list.append(pckh_2d_cam_act)
        if len(pckh_3d_cam_act):
            pckh_3d_cam_act_list.append(pckh_3d_cam_act)
        if len(mpjpe_cam_act):
            mpjpe_cam_act_list.append(mpjpe_cam_act)

    pckh_3d_list_aux = []
    for idx in range(len(pckh_3d_list[0])):
        pckh_3d_list_aux.append([pckh_3d[idx] for pckh_3d in pckh_3d_list])
    plot_multiple_pckh(pckh_3d_list_aux[0], pckh_3d_list_aux[1], pckh_3d_list_aux[2], pckh_3d_list_aux[3], test_ids, global_dir, flag="3D")

    build_results_tables(
        pckh_2d_list, pckh_3d_list, mpjpe_list, pckh_2d_cam_act_list, pckh_3d_cam_act_list, mpjpe_cam_act_list,
        test_ids_pckh_2d, test_ids_pckh_3d, test_ids_mpjpe
    )