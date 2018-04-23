#!/usr/bin/env python

"""
gen_subsets.py: Generate a subset with processed samples and labels
in HDF5 file format.
"""

import argparse
import glob
import math
import os
import scipy.io as sio
import sys
import time
from matplotlib import pyplot as plt
from random import shuffle

import cv2
import h5py
import numpy as np

__author__ = "David Pascual Hernandez"
__date__ = "2018/03/08"


def get_args():
    """
    Get program arguments and parse them.
    @return: dict - arguments
    """
    ap = argparse.ArgumentParser(description="Generate train, validation and "
                                             "test subsets from MPII and LSP "
                                             "datasets.")
    ap.add_argument("-p", "--pcts", type=int, required=True, nargs="+",
                    help="Percentage of samples for each subset")
    ap.add_argument("-d", "--data", type=str, required=False,
                    default="./", help="Datasets path")
    ap.add_argument("-b", "--boxsize", type=int, required=False,
                    default=368, help="samples size")
    ap.add_argument("-v", "--viz", type=bool, required=False,
                    default=False)

    a = vars(ap.parse_args())
    if len(a["pcts"]) != 3:
        ap.print_help()
        print("\nYou must specify pct. of samples for train, val and test !!")
        exit()
    if np.sum(a["pcts"]) != 100:
        ap.print_help()
        print("\nPercentages of samples for each subset must sum up to 100 !!")
        exit()

    return a


def updt(total, progress):
    """
    Displays or updates a console progress bar.

    Original source: https://stackoverflow.com/a/15860757/1391441
    """
    bar_length, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(bar_length * progress))
    text = "\t\r[{}] {:.0f}% {}".format(
        "#" * block + "-" * (bar_length - block), round(progress * 100, 0),
        status)
    sys.stdout.write(text)
    sys.stdout.flush()


def get_paths(base_path):
    """
    Get datasets paths and check whether or not they exist.
    @param base_path: str - base path
    @return: tuple, tuple - labels and images paths for each dataset
    """
    mpii = base_path + "MPII/"
    if not os.path.exists(mpii):
        print(mpii + " not found!")
        exit()
    mpii = (mpii + "images/", mpii + "mpii_human_pose_v1_u12_1.mat")

    lsp = base_path + "LSP/"
    if not os.path.exists(lsp):
        print(lsp + " not found!")
        exit()
    lsp = (lsp + "images/", lsp + "joints.mat")

    return mpii, lsp


def parse_mpii(mat):
    """
    Parse MPII .mat annotation file
    @param mat: dict - MPII annotation file
    @return: dict - parsed annotations
    """
    mat = mat["RELEASE"]
    annolist = mat["annolist"][0][0][0]

    is_train = mat["img_train"][0][0][0]
    ann_ims = [str(ann_im[0][0][0][0]) for ann_im in annolist["image"]]
    ann_rects = [ann_rect for ann_rect in annolist["annorect"]]

    parsed = {"is_train": is_train, "fname": ann_ims, "rects": ann_rects}

    return parsed


def avoid_mpii_test(labels):
    """
    Avoid including in the subsets test samples from the MPII dataset
    (they are not labeled)
    @param labels: dict - MPII labels
    @return: list - excluded samples
    """
    mpii_test = []
    for fname, is_train in zip(labels["fname"], labels["is_train"]):
        if not is_train:
            mpii_test.append(fname)

    return mpii_test


def choose_samples(path_mpii, path_lsp, pcts, mpii_excluded):
    """
    Choose random samples of every dataset to fill train, val and
    test subsets, keeping the same ratio of MPII/LSP samples in each
    one.
    @param path_mpii: str - MPII images and labels path
    @param path_lsp: str - LSP images and labels path
    @param pcts: list - Percentage of samples for each subset
    @param mpii_excluded: list - test MPII excluded samples
    @return: list, list, list - IDs of the samples assigned to
    each subset
    """
    # Get each image filename and shuffle them
    mpii_fnames = []
    for f in glob.glob(path_mpii + "*.jpg"):
        name = f.split("/")[-1]
        if name not in mpii_excluded:
            mpii_fnames.append(f)

    lsp_fnames = glob.glob(path_lsp + "*.jpg")

    shuffle(mpii_fnames), shuffle(lsp_fnames)

    bounds = [pcts[0] / 100., (pcts[0] + pcts[1]) / 100.]

    # MPII: divide in subsets
    mpii_bounds = [int(len(mpii_fnames) * b) for b in bounds]

    mpii_train = mpii_fnames[:mpii_bounds[0]]
    mpii_val = mpii_fnames[mpii_bounds[0]:mpii_bounds[1]]
    mpii_test = mpii_fnames[mpii_bounds[1]:]

    # LSP: divide in subsets and add dataset name to samples list
    lsp_bounds = [int(len(lsp_fnames) * b) for b in bounds]

    lsp_train = lsp_fnames[:lsp_bounds[0]]
    lsp_val = lsp_fnames[lsp_bounds[0]:lsp_bounds[1]]
    lsp_test = lsp_fnames[lsp_bounds[1]:]

    # Merge both datasets subsets and re-shuffle
    train = mpii_train + lsp_train
    val = mpii_val + lsp_val
    test = mpii_test + lsp_test

    shuffle(train), shuffle(val), shuffle(test)

    return train, val, test


def get_human(sample, c, s, bsize):
    """
    Crop human in the image depending on subject center and scale.
    @param sample: np.array - input image
    @param c: list - approx. human center
    @param s: float - approx. human scale wrt 200px
    @param bsize: int - boxsize
    @return: np.array - cropped human
    """
    cx, cy = c

    # Resize image and center according to given scale
    im_resized = cv2.resize(sample, None, fx=s, fy=s)

    h, w, d = im_resized.shape

    pad_up = int(bsize / 2 - cy)
    pad_down = int(bsize / 2 - (h - cy))
    pad_left = int(bsize / 2 - cx)
    pad_right = int(bsize / 2 - (w - cx))

    # Apply padding or crop image as needed
    if pad_up > 0:
        pad = np.ones((pad_up, w, d)) * 128
        im_resized = np.vstack((pad, im_resized))
    else:
        im_resized = im_resized[-pad_up:, :, :]
    h, w, d = im_resized.shape

    if pad_down > 0:
        pad = np.ones((pad_down, w, d)) * 128
        im_resized = np.vstack((im_resized, pad))
    else:
        im_resized = im_resized[:h + pad_down, :, :]
    h, w, d = im_resized.shape

    if pad_left > 0:
        pad = np.ones((h, pad_left, d)) * 128
        im_resized = np.hstack((pad, im_resized))
    else:
        im_resized = im_resized[:, -pad_left:, :]
    h, w, d = im_resized.shape

    if pad_right > 0:
        pad = np.ones((h, pad_right, d)) * 128
        im_resized = np.hstack((im_resized, pad))
    else:
        im_resized = im_resized[:, :w + pad_right, :]

    return im_resized


def count_humans(im_paths, mpii, lsp):
    """
    Count total number of humans in MPII and LSP samples.
    @param im_paths: list - images paths
    @param mpii: dict - MPII annotations
    @param lsp: dict - LSP annotations
    @return: int - number of humans
    """
    total_humans = 0
    for i, path in enumerate(im_paths):
        dataset, _, im_name = path.split("/")[-3:]
        if dataset == "MPII":
            idx = np.where(np.array(mpii["fname"]) == im_name)
            rects = mpii["rects"][int(idx[0])]
            if len(rects):
                total_humans += len(rects[0])
        if dataset == "LSP":
            total_humans += 1

    return total_humans


def create_subset(f, im_paths, annotations, bsize, joint_names, visualize):
    """
    Create a subset from MPII and LSP datasets.
    @param f: HDF5 file object
    @param im_paths: list - images paths
    @param annotations: dict - MPII and LSP annotations
    @param bsize: int - boxsize
    @param joint_names: list - joint names
    @param visualize: bool - visualization flag
    @return: filled HDF5 file object
    """
    mpii, lsp, lsp_heads = annotations

    total_humans = count_humans(im_paths, mpii, lsp)

    # Create and open labels/images subsets
    im_shape = (total_humans, bsize, bsize, 3)
    images = f.create_dataset("images", im_shape, np.uint8)

    labels_type = np.dtype([("dataset", np.str_, 3),
                            ("fname", np.str_, 13),
                            ("scale", np.float16, 1),
                            ("center", np.uint16, 2),
                            ("joints", np.uint16, (16, 2)),
                            ("headsize", np.float16, 1)])

    labels = f.create_dataset("labels", (total_humans,), labels_type)

    failed = 0
    n_humans = 0

    for i, path in enumerate(im_paths):
        im = cv2.imread(path)

        dataset, _, im_name = path.split("/")[-3:]

        """
        MPII Annotations and Images
        """
        if dataset == "MPII":
            idx = int(np.where(np.array(mpii["fname"]) == im_name)[0])

            if visualize:
                print("\nMPII image " + im_name)
                plt.figure(), plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                plt.figure()

            rects = mpii["rects"][idx]
            if len(rects):
                for j, rect in enumerate(rects[0]):
                    scale = 0
                    center = np.zeros(2)
                    joints = np.zeros((16, 2))
                    im_human = np.zeros((bsize, bsize, 3))
                    head_size = 0
                    try:
                        scale = float(rect["scale"])
                        center = rect["objpos"][0][0]

                        # Get scale
                        target_dist = 41. / 35.  # Mysterious parameter?
                        scale = target_dist / scale

                        # Get center
                        cx, cy = rect["objpos"][0][0]
                        center = np.array([int(cx * scale), int(cy * scale)])

                        # Get image
                        im_human = get_human(im, center, scale, bsize)

                        # Get joints
                        x_list = rect["annopoints"]["point"][0][0]["x"][0]
                        y_list = rect["annopoints"]["point"][0][0]["y"][0]
                        id_list = rect["annopoints"]["point"][0][0]["id"][0]

                        for x, y, j_id in zip(x_list, y_list, id_list):
                            joints[int(j_id[0])] = (int(x[0]), int(y[0]))

                        # Get head size
                        x2, y2 = (rect["x2"], rect["y2"])
                        x1, y1 = (rect["x1"], rect["y1"])
                        ssd = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        head_size = 0.6 * ssd

                        # Print scale/center and draw joints over image
                        if visualize:
                            print("Human " + str(j) + ":")

                            print("\tScale: " + str(scale))
                            print("\tCenter: " + str(center))

                            print("\tJoints:")
                            for j_id, joint in enumerate(joints):
                                print("\t\t" + joint_names[j_id] + ": "
                                      + str(joint))

                            print("\tHead size: " + str(head_size))

                            x1 = int((x1 * scale) - center[0] + (bsize / 2))
                            x2 = int((x2 * scale) - center[0] + (bsize / 2))
                            y1 = int((y1 * scale) - center[1] + (bsize / 2))
                            y2 = int((y2 * scale) - center[1] + (bsize / 2))

                            cv2.rectangle(im_human, (x2, y2), (x1, y1),
                                          (255, 0, 0), thickness=2)

                            plt.subplot(1, len(rects[0]), j + 1)
                            plt.imshow(cv2.cvtColor(np.uint8(im_human),
                                                    cv2.COLOR_BGR2RGB))

                            for x, y in joints:
                                x = (x * scale) - center[0] + (bsize / 2)
                                y = (y * scale) - center[1] + (bsize / 2)
                                plt.plot(x, y, "y*", markersize=bsize / 30)

                        # Store human data in created subset
                        data = np.array(("mpi", im_name, scale, center, joints,
                                         head_size), dtype=labels_type)
                        labels[n_humans] = data
                        images[n_humans] = im_human

                        n_humans += 1

                    except (ValueError, TypeError):
                        failed += 1

            else:
                failed += 1

            if visualize:
                plt.show()

        """
        LSP Annotations and Images
        """
        if dataset == "LSP":
            idx = im_name[2:6]

            # Get scale
            scale = im.shape[0] / float(boxsize)
            target_dist = 0.8  # Mysterious parameter?
            scale = target_dist / scale

            # Get center
            cx, cy = (im.shape[1] / 2, im.shape[0] / 2)
            center = np.array([int(cx * scale), int(cy * scale)])

            # Get image
            im_human = get_human(im, center, scale, bsize)

            # Get joints
            rect = lsp[:, :, int(idx) - 1]
            joints = np.zeros((16, 2))
            id_list = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 8, 9]
            for x, y, j_id in zip(rect[0, :], rect[1, :], id_list):
                joints[j_id] = (int(x), int(y))

            # Get head size
            x1, y1, x2, y2 = lsp_heads[idx]
            ssd = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            head_size = 0.6 * ssd

            # Print scale/center and draw joints over image
            if visualize:
                print("\nImage " + im_name + " ---- " + idx)

                print("\tScale: " + str(scale))
                print("\tCenter: " + str(center))

                print("\tJoints:")
                for j_id, joint in enumerate(joints):
                    print("\t\t" + joint_names[j_id] + ": " + str(joint))

                print("\tHead size: " + str(head_size))

                plt.figure(), plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

                x1 = int((x1 * scale) - center[0] + (bsize / 2))
                x2 = int((x2 * scale) - center[0] + (bsize / 2))
                y1 = int((y1 * scale) - center[1] + (bsize / 2))
                y2 = int((y2 * scale) - center[1] + (bsize / 2))

                cv2.rectangle(im_human, (x2, y2), (x1, y1),
                              (255, 0, 0), thickness=2)

                plt.figure()
                plt.imshow(cv2.cvtColor(np.uint8(im_human), cv2.COLOR_BGR2RGB))
                for x, y in joints:
                    x = (x * scale) - center[0] + (bsize / 2)
                    y = (y * scale) - center[1] + (bsize / 2)

                    plt.plot(x, y, "y*", markersize=bsize / 30)

                plt.xlim((0, bsize))
                plt.ylim((bsize, 0))

                plt.show()

            # Store human data in created subet
            data = np.array(("LSP", im_name, scale, center, joints, head_size),
                            dtype=labels_type)
            labels[n_humans] = data
            images[n_humans] = im_human

            n_humans += 1

        updt(total_humans, n_humans)

    # print("\tHumans total/failed: " + str(n_humans) + "/" + str(failed))

    return f


if __name__ == "__main__":
    args = get_args()
    viz = args["viz"]
    data_path = args["data"]
    boxsize = args["boxsize"]
    samples_pcts = args["pcts"]

    # Get datasets paths
    (mpii_images, mpii_labels), (lsp_images, lsp_labels) = get_paths(data_path)

    # Read .mat labels files into dictionaries
    mpii_labels = parse_mpii(sio.loadmat(mpii_labels))
    lsp_labels = sio.loadmat(lsp_labels)["joints"][:2]
    lsp_heads = np.load(data_path + "LSP/anno_head.npy").item()
    all_labels = (mpii_labels, lsp_labels, lsp_heads)

    # Choose which samples will be assigned to each subset
    excluded = avoid_mpii_test(mpii_labels)  # test MPII images

    fnames = choose_samples(mpii_images, lsp_images, samples_pcts, excluded)
    im_train, im_val, im_test = fnames

    if viz:
        print("\n\n# samples of each subset:")
        print(len(im_train), len(im_val), len(im_test))

        print("Random samples:")
        for index in range(5):
            print("\tTraining: " + str(im_train[index]))
            print("\tValidation: " + str(im_val[index]))
            print("\tTest: " + str(im_test[index]))
        time.sleep(5)

    # Create .h5 files for each subset
    ftrain = h5py.File("mpii_lsp_train.h5", mode="w")
    fval = h5py.File("mpii_lsp_val.h5", mode="w")
    ftest = h5py.File("mpii_lsp_test.h5", mode="w")

    # Labeled joint names
    jnames = ["r_ankle", "r_knee", "r_hip", "l_hip", "l_knee", "l_ankle",
              "pelvis", "thorax", "neck", "head_top", "r_wrist", "r_elbow",
              "r_shoulder", "l_shoulder", "l_elbow", "l_wrist"]

    # Fill subsets
    print("\nCreating training subset...")
    ftrain = create_subset(ftrain, im_train, all_labels, boxsize, jnames, viz)
    ftrain.close()

    print("\nCreating validation subset...")
    fval = create_subset(fval, im_val, all_labels, boxsize, jnames, viz)
    fval.close()

    print("\nCreating test subset...")
    ftest = create_subset(ftest, im_test, all_labels, boxsize, jnames, viz)
    ftest.close()
