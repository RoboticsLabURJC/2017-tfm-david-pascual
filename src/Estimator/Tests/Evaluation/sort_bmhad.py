from glob import glob
import os
from shutil import copy2

if __name__ == "__main__":
    input_dir = "/media/dpascualhe/Elements/bmhad"
    output_dir = "/home/dpascualhe/repos/2017-tfm-david-pascual/src/Estimator/Tests/Datasets/BMHAD"

    num_subjects = 12
    num_actions = 11
    num_repetitions = 2

    for s in range(num_subjects):
        for a in range(num_actions):
            for r in range(1, num_repetitions):
                s += 1
                a += 1
                r += 1
                print("Copying s%02da%02dr%02dk01" % (s, a, r))

                scene_output_dir_k01 = os.path.join(output_dir, "s%02da%02dr%02dk01" % (s, a, r))
                if not os.path.isdir(scene_output_dir_k01):
                    os.mkdir(scene_output_dir_k01)
                video_output_dir_k01 = os.path.join(scene_output_dir_k01, "video")
                if not os.path.isdir(video_output_dir_k01):
                    os.mkdir(video_output_dir_k01)
                    
                scene_output_dir_k02 = os.path.join(output_dir, "s%02da%02dr%02dk02" % (s, a, r))
                if not os.path.isdir(scene_output_dir_k02):
                    os.mkdir(scene_output_dir_k02)
                video_output_dir_k02 = os.path.join(scene_output_dir_k02, "video")
                if not os.path.isdir(video_output_dir_k02):
                    os.mkdir(video_output_dir_k02)
                    
                correspondences_k01 = glob(os.path.join(input_dir, "Correspondences", "*kin01_s%02d_a%02d_r%02d*" % (s, a, r)))[0]
                correspondences_k02 = glob(os.path.join(input_dir, "Correspondences", "*kin02_s%02d_a%02d_r%02d*" % (s, a, r)))[0]

                skeleton = glob(os.path.join(input_dir, "SkeletalData", "*_s%02d_a%02d_r%02d*" % (s, a, r)))[0]

                video_k01 = os.path.join(input_dir, "Kin01", "S%02d" % s, "A%02d" % a, "R%02d" % r)
                video_k02 = os.path.join(input_dir, "Kin02", "S%02d" % s, "A%02d" % a, "R%02d" % r)

                copy2(correspondences_k01, scene_output_dir_k01)
                copy2(correspondences_k02, scene_output_dir_k02)

                copy2(skeleton, scene_output_dir_k01)
                copy2(skeleton, scene_output_dir_k02)

                for file in glob(os.path.join(video_k01, "kin*")):
                    copy2(file, video_output_dir_k01)

                for file in glob(os.path.join(video_k02, "kin*")):
                    copy2(file, video_output_dir_k02)