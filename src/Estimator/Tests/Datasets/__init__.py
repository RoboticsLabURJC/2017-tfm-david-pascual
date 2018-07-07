#!/usr/bin/env python

"""
foo.py: bar.
"""

import argparse
import cv2
import math
import numpy as np
import time
from matplotlib import pyplot as plt

__author__ = "David Pascual Hernandez"
__date__ = "2018/mm/dd"


def get_args():
    """
    Get program arguments and parse them.
    :return: dict - arguments
    """
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("-", "--", type=str, required=True, help="")

    return vars(ap.parse_args())


if __name__ == "__main__":
    args = get_args()
