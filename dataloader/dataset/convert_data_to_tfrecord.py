# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys
import xml.etree.cElementTree as ET
import numpy as np
import tensorflow as tf
import glob
import cv2
import os
from pktool import rovoc_parse, thetaobb2pointobb, mkdir_or_exist,simpletxt_parse


sys.path.append('../../')

# from libs.label_name_dict.label_dict import LabelMap
# from utils.tools import makedirs, view_bar
# from libs.configs import cfgs

tf.app.flags.DEFINE_string('VOC_dir', 'F:/data/gf2_v2/trainval/dota/', 'Voc dir')
tf.app.flags.DEFINE_string('xml_dir', 'sublabel', 'xml dir')
tf.app.flags.DEFINE_string('image_dir', 'subimg', 'image dir')
tf.app.flags.DEFINE_string('save_name', 'train', 'save name')
tf.app.flags.DEFINE_string('save_dir', '../tfrecord/', 'save name')
tf.app.flags.DEFINE_string('img_format', '.png', 'format of image')
tf.app.flags.DEFINE_string('dataset', 'DOTA2.0', 'dataset')
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
    xml_path = os.path.join(FLAGS.VOC_dir, FLAGS.xml_dir)
    image_path = os.path.join(FLAGS.VOC_dir, FLAGS.image_dir)
    print(image_path)
    save_path = os.path.join(FLAGS.save_dir, FLAGS.dataset + '_' + FLAGS.save_name + '.tfrecord')
    mkdir_or_exist(FLAGS.save_dir)

    # writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    # writer = tf.python_io.TFRecordWriter(path=save_path, options=writer_options)
    writer = tf.python_io.TFRecordWriter(path=save_path)
    for count, xml in enumerate(glob.glob(xml_path + '/*.txt')):

        img_name = xml.split('\\')[-1].split('.txt')[0] + FLAGS.img_format
        img_path = image_path + '/' + img_name

        if not os.path.exists(img_path):
            print('{} is not exist!'.format(img_path))
            continue
        ships = simpletxt_parse(xml,space=' ',boxType='points')
        gtboxes_and_label=[]
        print(len(ships))
        for ship in ships:
            gtbox_label=[0,0,0,0,0,0,0,0,1]
            gtbox_label[:8]=ship['points']
            gtboxes_and_label.append(gtbox_label)
        img_height, img_width=800,800
        gtboxes_and_label=np.array(gtboxes_and_label, dtype=np.int32)
        # if img_height != 600 or img_width != 600:
        #     continue

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
        # img_raw = img.tobytes()
        # label = gtboxes_and_label.shape[0]
        # shape = gtboxes_and_label.shape
        # # gtboxes_and_label=gtboxes_and_label.tobytes()
        # example = tf.train.Example(features=tf.train.Features(feature={
        #         'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        #         'img_name': _bytes_feature(img_name.encode()),
        #         'img_width': _int64_feature(img_width),
        #         'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        #         'gtboxes_and_label':_bytes_feature(gtboxes_and_label.tostring())
        #         }))

        writer.write(example.SerializeToString())

        # view_bar('Conversion progress', count + 1, len(glob.glob(xml_path + '/*.xml')))

    print('\nConversion is complete!')
    writer.close()


if __name__ == '__main__':
    # xml_path = '../data/dataset/VOCdevkit/VOC2007/Annotations/000005.xml'
    # read_xml_gtbox_and_label(xml_path)

    convert_pascal_to_tfrecord()
