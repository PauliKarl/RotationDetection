# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo

"""
This is your result for task 1:

    mAP: 0.7066194189913816
    ap of each class:
    plane:0.8905480010393588,
    baseball-diamond:0.7845764249543027,
    bridge:0.4415489914209597,
    ground-track-field:0.6515721505439082,
    small-vehicle:0.7509226622459368,
    large-vehicle:0.7288453788151275,
    ship:0.8604046905135039,
    tennis-court:0.9082569687774237,
    basketball-court:0.8141347275878138,
    storage-tank:0.8253027715641935,
    soccer-ball-field:0.5623560181901192,
    roundabout:0.6100656068973895,
    harbor:0.5648618127447264,
    swimming-pool:0.6767393616949172,
    helicopter:0.5291557178810407

The submitted information is :

Description: RetinaNet_DOTA_R3Det_2x_20191108_70.2w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue


"""
DATASET_VERSION = 'v1'
SDC_TYPE = 'shipdet'
CLASSES_NUM =1
###load datset sdc-multidet
# DATASET_VERSION = 'v0'
# SDC_TYPE = 'multidet'
# CLASSES_NUM =16
# ------------------------------------------------
VERSION = 'FPN_Res50_r3det_1x_20210405'
NET_NAME = 'resnet_v1_50'

# ---------------------------------------- System
# ROOT_PATH = os.path.abspath('../../')
# print(20*"++--")
# print(ROOT_PATH)
# GPU_GROUP = "0,1,2"
# NUM_GPU = len(GPU_GROUP.strip().split(','))
# SHOW_TRAIN_INFO_INTE = 20
# SMRY_ITER = 200
# SAVE_WEIGHTS_INTE = 27000 * 2
ROOT_PATH=os.path.abspath('../../')
work_PATH = '/data2/pd/sdc/shipdet/v1/works_dir/rodet/'
print(20*"++--")
print(ROOT_PATH)
print(work_PATH)
GPU_GROUP = '1'
NUM_GPU = len(GPU_GROUP.strip().split(','))
SHOW_TRAIN_INFO_INTE = 20
SMRY_ITER = 200
SAVE_WEIGHTS_INTE = 3500*2 ##6886/BATCH_SIZE



SUMMARY_PATH = os.path.join(work_PATH, 'output/summary')
TEST_SAVE_PATH = os.path.join(work_PATH, 'tools/test_result')

pretrain_zoo = PretrainModelZoo()
PRETRAINED_CKPT = pretrain_zoo.pretrain_weight_path(NET_NAME, ROOT_PATH)
TRAINED_CKPT = os.path.join(work_PATH, 'output/trained_weights')
EVALUATE_R_DIR = os.path.join(work_PATH, 'output/evaluate_result_pickle/')

# ------------------------------------------ Train and test
RESTORE_FROM_RPN = False
FIXED_BLOCKS = 1  # allow 0~3
FREEZE_BLOCKS = [True, False, False, False, False]  # for gluoncv backbone
USE_07_METRIC = True
ADD_BOX_IN_TENSORBOARD = True

MUTILPY_BIAS_GRADIENT = 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = 10.0  # if None, will not clip

CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0
USE_IOU_FACTOR = False

BATCH_SIZE = 1
EPSILON = 1e-5
MOMENTUM = 0.9
LR = 5e-4
DECAY_STEP = [SAVE_WEIGHTS_INTE*12, SAVE_WEIGHTS_INTE*16, SAVE_WEIGHTS_INTE*20]
MAX_ITERATION = SAVE_WEIGHTS_INTE*20
WARM_SETP = int(1.0 / 4.0 * SAVE_WEIGHTS_INTE)

# -------------------------------------------- Dataset
DATASET_NAME = 'sdc'
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
PIXEL_MEAN_ = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
IMG_SHORT_SIDE_LEN = 1024
IMG_MAX_LENGTH = 1024
CLASS_NUM = CLASSES_NUM

IMG_ROTATE = False
RGB2GRAY = False
VERTICAL_FLIP = False
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = False

# --------------------------------------------- Network
SUBNETS_WEIGHTS_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
SUBNETS_BIAS_INITIALIZER = tf.constant_initializer(value=0.0)
PROBABILITY = 0.01
FINAL_CONV_BIAS_INITIALIZER = tf.constant_initializer(value=-math.log((1.0 - PROBABILITY) / PROBABILITY))
WEIGHT_DECAY = 1e-4
USE_GN = False
NUM_SUBNET_CONV = 4
NUM_REFINE_STAGE = 1
USE_RELU = False
FPN_CHANNEL = 256
FPN_MODE = 'fpn'

# --------------------------------------------- Anchor
LEVEL = ['P3', 'P4', 'P5', 'P6', 'P7']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
ANCHOR_STRIDE = [8, 16, 32, 64, 128]
ANCHOR_SCALES = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
ANCHOR_RATIOS = [1, 1 / 2, 2., 1 / 3., 3., 5., 1 / 5.]
ANCHOR_ANGLES = [-90, -75, -60, -45, -30, -15]
ANCHOR_SCALE_FACTORS = None
USE_CENTER_OFFSET = True
METHOD = 'R'
USE_ANGLE_COND = False
ANGLE_RANGE = 90

# -------------------------------------------- Head
SHARE_NET = True
USE_P5 = True
IOU_POSITIVE_THRESHOLD = 0.5
IOU_NEGATIVE_THRESHOLD = 0.4
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

NMS = True
NMS_IOU_THRESHOLD = 0.1
MAXIMUM_DETECTIONS = 100
FILTERED_SCORE = 0.05
VIS_SCORE = 0.4

