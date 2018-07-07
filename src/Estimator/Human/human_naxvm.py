#!/usr/bin/env python

"""
human_naxvm.py: Human detection with @naxvm model.
"""

__author__ = "David Pascual Hernandez"
__date__ = "2018/05/22"

import numpy as np
import tensorflow as tf

from human import Human
from utils.naxvm import label_map_util

LABELS_DICT = {'voc': 'models/naxvm/labels/pascal_label_map.pbtxt',
               'coco': '/home/dpascualhe/PycharmProjects/2017-tfm-david-pascual/src/Estimator/Human/models/naxvm/labels/mscoco_label_map.pbtxt',
               'kitti': 'models/naxvm/labels/kitti_label_map.txt',
               'oid': 'models/naxvm/labels/oid_bboc_trainable_label_map.pbtxt',
               'pet': 'models/naxvm/labels/pet_label_map.pbtxt'}

DB = "coco"

class HumanDetector(Human):
    """
    Class for person detection.
    """

    def __init__(self, model, boxsize=192):
        """
        Class constructor.
        @param model: tf models
        @param weights: tf models weights
        """
        Human.__init__(self, boxsize)  # boxsize will not be used in this case!

        self.config = tf.ConfigProto(device_count={"GPU": 1},
                                     allow_soft_placement=True,
                                     log_device_placement=False)
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.5

        labels_file = LABELS_DICT[DB]
        lbl_map = label_map_util.load_labelmap(labels_file) # loads the labels map.
        categories = label_map_util.convert_label_map_to_categories(lbl_map, 9999)
        category_index = label_map_util.create_category_index(categories)

        self.classes = {}
        # We build is as a dict because of gaps on the labels definitions
        for cat in category_index:
            self.classes[cat] = str(category_index[cat]['name'])

        # Frozen inference graph, written on the file
        CKPT = model
        detection_graph = tf.Graph() # new graph instance.
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        self.sess = tf.Session(graph=detection_graph, config=self.config)
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # NCHW conversion. not possible
        #self.image_tensor = tf.transpose(self.image_tensor, [0, 3, 1, 2])
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        self.boxes = []
        self.scores = []
        self.predictions = []


        # Dummy initialization (otherwise it takes longer then)
        dummy_tensor = np.zeros((1,1,1,3), dtype=np.int32)
        self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: dummy_tensor})

        self.confidence_threshold = 0.5

    def detect(self):
        """
        Detects people in the image.
        @param im: np.array - input image
        @return: np.array - heatmap
        """
        image_np_expanded = np.expand_dims(self.im, axis=0)

        graph_elements = [self.detection_boxes, self.detection_scores,
                          self.detection_classes, self.num_detections]
        input_dict = {self.image_tensor: image_np_expanded}

        return self.sess.run(graph_elements, feed_dict=input_dict)

    def get_bboxes(self, im):
        self.im = im
        h, w, d = self.im.shape

        (boxes, scores, predictions, _) = self.detect()

        # We only keep the most confident predictions.
        conf = scores > self.confidence_threshold # bool array
        boxes = boxes[conf]

        # aux variable for avoiding race condition while int casting
        tmp_boxes = np.zeros([len(boxes), 4])
        tmp_boxes[:,[0,2]] = boxes[:,[1,3]] * w
        tmp_boxes[:,[3,1]] = boxes[:,[2,0]] * h

        bboxes = []
        for tmp_box in tmp_boxes:
            upper_left = tuple(tmp_box[:2])
            bottom_right = tuple(tmp_box[2:])
            bboxes.append((upper_left, bottom_right))

        return np.array(bboxes, dtype=np.int64)