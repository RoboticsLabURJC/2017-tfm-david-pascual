#!/usr/bin/env python

"""
parse_anno_head.py: Parse XMLs containing head bounding box for each
LSP image.
"""

import argparse
import glob
import math
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
import xmltodict

__author__ = "David Pascual Hernandez"
__date__ = "2018/04/08"


def get_args():
    """
    Get program arguments and parse them.
    :return: dict - arguments
    """
    ap = argparse.ArgumentParser(description="Parse head annotation XMLs")
    ap.add_argument("-p", "--path", type=str, required=True, help="XMLs path")

    return vars(ap.parse_args())


if __name__ == "__main__":
    args = get_args()
    path = args["path"]

    new_anno = {}
    for xml_anno in glob.glob(path + "*.xml"):
        with open(xml_anno) as f:
            xml_dict = xmltodict.parse(f.read())

            im_id = xml_dict["annotation"]["filename"][2:6]

            if type(xml_dict["annotation"]["object"]) == list:
                for rect in xml_dict["annotation"]["object"]:
                    if not rect["deleted"]:
                        bbox = rect["polygon"]["pt"]
            else:
                bbox = xml_dict["annotation"]["object"]["polygon"]["pt"]

            x1, y1 = bbox[0].values()
            x2, y2 = bbox[2].values()

            new_anno[str(im_id)] = [int(x1), int(y1), int(x2), int(y2)]

    np.save("anno_head.npy", new_anno)
