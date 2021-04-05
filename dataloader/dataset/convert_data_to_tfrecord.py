# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys
import xml.etree.cElementTree as ET
import numpy as np
import tensorflow as tf
import glob
import cv2
import os
from pktool import rovoc_parse, thetaobb2pointobb, mkdir_or_exist,simpletxt_parse,get_files


sys.path.append('../../')

from libs.label_name_dict.label_dict import LabelMap
# from utils.tools import makedirs, view_bar
from libs.configs import cfgs
# 'sdc','sdc-multidet'
dataset_Name = 'sdc-multidet'
cfgs.DATASET_NAME = dataset_Name

tf.app.flags.DEFINE_string('VOC_dir', '/data2/pd/sdc/multidet/v0/', 'Voc dir')
tf.app.flags.DEFINE_string('txt_dir', 'labels', 'xml dir')
tf.app.flags.DEFINE_string('image_dir', 'images', 'image dir')
tf.app.flags.DEFINE_string('save_dir', '/data2/pd/sdc/multidet/v0/tfrecord/', 'save name')
tf.app.flags.DEFINE_string('img_format', '.png', 'format of image')
tf.app.flags.DEFINE_string('dataset', dataset_Name, 'dataset')
FLAGS = tf.app.flags.FLAGS

def int64_feature(values):
    if not isinstance(values,(tuple,list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_xml_gtbox_and_label(xml_path):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 9],
           and has [x1, y1, x2, y2, x3, y3, x4, y4, label] in a per row
    """
    # label_map = LabelMap(cfgs)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    for child_of_root in root:
        # if child_of_root.tag == 'filename':
        #     assert child_of_root.text == xml_path.split('/')[-1].split('.')[0] \
        #                                  + FLAGS.img_format, 'xml_name and img_name cannot match'

        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)

        if child_of_root.tag == 'object':
            label = None
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    label = 1#label_map.name2label()[child_item.text]
                if child_item.tag == 'bndbox':
                    tmp_box = []
                    for node in child_item:
                        tmp_box.append(float(node.text))
                    assert label is not None, 'label is none, error'
                    tmp_box.append(label)
                    box_list.append(tmp_box)

    gtbox_label = np.array(box_list, dtype=np.int32)

    return img_height, img_width, gtbox_label


def convert_pascal_to_tfrecord():
    """convert txt (points + label format) to tfrecord
        VOC_dir:
            --trainval
                --images
                --labels
            --test
                --images
                --labels
    """
    allNeedConvert = ['trainval','test']
    for train_or_test in allNeedConvert:

        label_Path = FLAGS.VOC_dir + "{}/".format(train_or_test)+ FLAGS.txt_dir
        image_path = FLAGS.VOC_dir + "{}/".format(train_or_test)+ FLAGS.image_dir
        # xml_path = os.path.join(FLAGS.VOC_dir, FLAGS.xml_dir)
        # image_path = os.path.join(FLAGS.VOC_dir, FLAGS.image_dir)
        print(image_path)
        save_path = os.path.join(FLAGS.save_dir, FLAGS.dataset + '_' + train_or_test + '.tfrecord')
        mkdir_or_exist(FLAGS.save_dir)

        writer = tf.python_io.TFRecordWriter(path=save_path)

        txtFullPathList,_ = get_files(label_Path,_ends=['*.txt'])
        for count, txt in enumerate(txtFullPathList):
            (txtPath,tmpTxtName) = os.path.split(txt)
            (txt_name,extension) = os.path.splitext(tmpTxtName)

            img_name = txt_name + FLAGS.img_format
            img_path = image_path + '/' + img_name

            if not os.path.exists(img_path):
                print('{} is not exist!'.format(img_path))
                continue
            ships = simpletxt_parse(txt,space=' ',boxType='points')
            label_map = LabelMap(cfgs)
            # print(label_map.name2label())
            gtboxes_and_label=[]
            for ship in ships:
                gtbox_label=[0,0,0,0,0,0,0,0,0]
                gtbox_label[:8]=ship['points']
                gtbox_label[8] = label_map.name2label()[ship['label']]
                gtboxes_and_label.append(gtbox_label)
            img_height, img_width=1024,1024
            gtboxes_and_label=np.array(gtboxes_and_label, dtype=np.int32)

            img = cv2.imread(img_path)[:, :, ::-1]
            img=np.array(img, dtype=np.int32)
            img_raw = img.tobytes()
            num_objects = gtboxes_and_label.shape[0]
            # shape = gtboxes_and_label.shape
            # gtboxes_and_label=gtboxes_and_label.tobytes()
            feature = tf.train.Features(feature={
                # do not need encode() in linux
                'img_name': _bytes_feature(img_name.encode()),
                # 'img_name': _bytes_feature(img_name),
                'img_height': _int64_feature(img_height),
                'img_width': _int64_feature(img_width),
                'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'num_objects':tf.train.Feature(int64_list=tf.train.Int64List(value=[num_objects])),
                'gtboxes_and_label': _bytes_feature(gtboxes_and_label.tostring())
            })
            example = tf.train.Example(features=feature)

            writer.write(example.SerializeToString())

        print('Conversion is complete!save path:{}'.format(train_or_test,save_path))
        writer.close()


if __name__ == '__main__':

    convert_pascal_to_tfrecord()
