import os

import h5py
import numpy as np


def parse_sequence(subject, action, sequence, folder=""):
    anno_path = os.path.join(folder, "Subject%s_annotations/%s/%s.txt" % (subject, action, sequence))
    lines_anno = [line.rstrip('/n') for line in open(anno_path)]

    raw_frames_path = os.path.join(folder, "Subject%s_rgbd_rawtext/%s/%s_rgbd.txt" % (subject, action, sequence))
    lines_raw_frames = [line.rstrip('/n') for line in open(raw_frames_path)]

    data = []
    for line_anno, line_raw_frame in zip(lines_anno, lines_raw_frames):
        try:
            frame_anno = read_annotation_line(line_anno)
            raw_frame_rgb, raw_frame_depth = read_images_from_txt(line_raw_frame)
            frame_data = {"anno": frame_anno,
                          "rgb": raw_frame_rgb,
                          "depth": raw_frame_depth}
            data.append(frame_data)
        except ValueError:
            print("WARNING: We've reached end of file!")

    return data


def x_pixel_from_coords(x, y, z):
    k_realworld_x_to_z = 1.122133
    k_resolution_x = 640
    f_coeff_x = k_resolution_x / k_realworld_x_to_z

    return int((f_coeff_x * x / z) + (k_resolution_x / 2))


def y_pixel_from_coords(x, y, z):
    k_realworld_y_to_z = 0.84176
    k_resolution_y = 640
    f_coeff_x = k_resolution_y / k_realworld_y_to_z

    return int((f_coeff_x * x / z) + (k_resolution_y / 2))


def read_annotation_line(line):
    anno = {}
    raw_data = line.split(",")

    anno["frame_id"] = int(raw_data[0]),
    anno["joints"] = []

    joints_id = ["head", "neck", "torso", "left_shoulder", "left_elbow", "right_shoulder", "right_elbow", "left_hip",
                 "left_knee", "right_hip", "right_knee", "left_hand", "right_hand", "left_foot", "right_foot"]

    counter = 0
    for idx, joint_id in enumerate(joints_id):
        if idx < 11:
            anno_joint = {"joint_id": joint_id,
                          "orientation": np.array(raw_data[1 + counter: 10 + counter], np.float),
                          "orientation_conf": float(raw_data[10 + counter]),
                          "real_world_pos": np.array(raw_data[11 + counter: 14 + counter], np.float),
                          "pos_conf": float(raw_data[14 + counter])}
            counter += 14
        else:
            anno_joint = {"joint_id": joint_id,
                          "orientation": [] * 9,
                          "orientation_conf": 0,
                          "real_world_pos": np.array(raw_data[1 + counter: 4 + counter], np.float),
                          "pos_conf": float(raw_data[4 + counter])}
            counter += 4

        x_real, y_real, z_real = anno_joint["real_world_pos"]
        anno_joint["pixel_pos"] = np.array((x_pixel_from_coords(x_real, y_real, z_real)), np.int)
        anno["joints"].append(anno_joint)

    return anno


def read_images_from_txt(line):
    raw_data = line.split(",")[1:-1]  # strip frame number and last element, which is empty
    image_ravel = np.array(raw_data, np.uint16)
    image_rgbd = image_ravel.reshape((480, 640, 4))

    return image_rgbd[:, :, :3].astype(np.uint8), image_rgbd[:, :, 3]


def data_dict_to_hdf5(data, subject_id, action_id, sequence_id):
    rgb_data_shape = (len(data), 480, 640, 3)
    depth_data_shape = (len(data), 480, 640)
    anno_type = np.dtype([("joint_id", np.str, 15),
                          ("orientation", np.float16, 2),
                          ("orientation_conf", np.float16, 9),
                          ("real_world_pos", np.float16, 3),
                          ("pixel_pos", np.uint32, 2),
                          ("pixel_conf", np.float16, 1)])

    f = h5py.File("cad120-%s_%s_%s.h5" % (subject_id, action_id, sequence_id), mode="w")

    rgb = f.create_dataset("rgb", rgb_data_shape, np.uint8)
    depth = f.create_dataset("depth", depth_data_shape, np.uint16)
    labels = f.create_dataset("anno", (len(data), 14), anno_type)

    for idx, frame_data in enumerate(data):
        rgb[idx] = frame_data["rgb"]
        depth[idx] = frame_data["depth"]

        full_anno = frame_data["anno"]
        for idx_joint, anno in enumerate(full_anno["joints"]):
            labels[idx, idx_joint] = np.array((anno["joint_id"],
                                               anno["orientation"],
                                               anno["orientation_conf"],
                                               anno["real_world_pos"],
                                               anno["pixel_pos"],
                                               anno["pixel_conf"]), dtype=anno_type)

    f.close()


if __name__ == "__main__":
    subject_id = "1"
    action_id = "arranging_objects"
    sequence_id = "0510175411"

    global_folder = "/home/dpascualhe/repos/2017-tfm-david-pascual/src/Estimator/Tests/Datasets/CAD-120/"

    data = parse_sequence(subject_id, action_id, sequence_id, global_folder)
    data_dict_to_hdf5(data, subject_id, action_id, sequence_id)
