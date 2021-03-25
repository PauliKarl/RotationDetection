import os
import tensorflow as tf
import cv2

IMG_RESIZE = 800

def write_binary():  
    '''
    将默认路径下的所有图片存为TFRecord格式 保存到文件data.tfrecord中
    '''
    
    #创建对象 用于向记录文件写入记录
    writer = tf.python_io.TFRecordWriter('data.tfrecord')  
    
    dir_train = os.path.join(os.getcwd(),'..','kaggle_cv_train')
    classes = os.listdir(dir_train)[:1]
    print(classes)
    
    #遍历每一个子文件夹
    for index, name in enumerate(classes):
        class_path = os.path.join(dir_train,name)
        #遍历子目录下的每一个文件
        for img_name in os.listdir(class_path):
            #每一个图片全路径
            img_path = os.path.join(class_path , img_name)
            #读取图像
            img = cv2.imread(img_path)
            #缩放
            img1 = cv2.resize(img,(IMG_RESIZE,IMG_RESIZE))
            #将图片转化为原生bytes
            img_raw = img1.tobytes()         
            #将数据整理成 TFRecord 需要的数据结构 
            example = tf.train.Example(features=tf.train.Features(feature={
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index]))}))  

            #序列化  
            serialized = example.SerializeToString()  
            #写入文件  
            writer.write(serialized)  
    writer.close()
    
def read_and_decode(filename):  
    '''
    读取TFRecord格式格式文件，返回读取到的一张图像以及对应的标签 
    
    args:
        filename:TFRecord格式文件路径
              
    '''
    #创建文件队列,不限读取的数量  
    filename_queue = tf.train.string_input_producer([filename], shuffle=False, num_epochs=2)  
    #创建一个文件读取器 从队列文件中读取数据
    reader = tf.TFRecordReader()  
    
    #reader从 TFRecord 读取内容并保存到 serialized_example中 
    _, serialized_example = reader.read(filename_queue)  

    # 读取serialized_example的格式 
    # features = tf.parse_single_example(     
    #     serialized_example,  
    #     features={
    #         'img_raw': tf.FixedLenFeature([], tf.string),
    #         'img_name': tf.FixedLenFeature([], tf.string),
    #         'img_width': tf.FixedLenFeature([], tf.int64),
    #         'label': tf.FixedLenFeature([], tf.int64) ,
    #         'gtboxes_and_label': tf.FixedLenFeature([],tf.string)
    #     }  
    # )  
    features = tf.parse_single_example(
        serialized=serialized_example,
        features={
            'img_name': tf.FixedLenFeature([], tf.string),
            'img_height': tf.FixedLenFeature([], tf.int64),
            'img_width': tf.FixedLenFeature([], tf.int64),
            'img': tf.FixedLenFeature([], tf.string),
            'num_objects': tf.FixedLenFeature([], tf.int64),
            'gtboxes_and_label': tf.FixedLenFeature([], tf.string)
            
        }
    )
    # 解析从 serialized_example 读取到的内容  
    # img=tf.decode_raw(features['img_raw'],tf.uint8)
    # img = tf.reshape(img, [IMG_RESIZE, IMG_RESIZE, 3])
    # img_name = features['img_name']
    # img_width = features['img_width']
    # label = tf.cast(features['label'], tf.int32)

    # gtboxes_and_label = tf.decode_raw(features['gtboxes_and_label'], tf.uint8)
    # gt_shape = tf.stack([label,9])
    # gtboxes_and_label = tf.reshape(gtboxes_and_label,gt_shape)

    img_name = features['img_name']
    img_height = tf.cast(features['img_height'], tf.int32)
    img_width = tf.cast(features['img_width'], tf.int32)
    img = tf.decode_raw(features['img'], tf.int32)

    img = tf.reshape(img, shape=[img_height, img_width, 3])

    num_objects = tf.cast(features['num_objects'], tf.int32)

    gtboxes_and_label = tf.decode_raw(features['gtboxes_and_label'], tf.int32)
    gt_shape = tf.stack([num_objects,9])
    gtboxes_and_label = tf.reshape(gtboxes_and_label,gt_shape)

    
    return img,img_name,img_height,img_width,gtboxes_and_label,num_objects
    
#读取TFRecord格式格式文件，返回读取到的一张图像以及对应的标签
pattern = '/data2/pd/sdc/shipdet/tfrecord/sdc_trainval.tfrecord'

img,img_name,img_height,img_width,gtboxes_and_label,num_objects= read_and_decode(pattern)  

batch_size = 1

'''
读取批量数据 
'''
img,img_name,img_height,img_width,gtboxes_and_label,num_objects = tf.train.batch([img,img_name,img_height,img_width,gtboxes_and_label,num_objects], batch_size=batch_size, capacity=20, num_threads=2,dynamic_pad=True)

#顺序读取批量图片
# img_batch, label_batch = tf.train.batch([img,label], batch_size=batch_size, capacity=2000, num_threads=1)  
'''
tf.train.batch()
    tensors：一个列表或字典的tensor用来进行入队
    batch_size：设置每次从队列中获取出队数据的数量
    num_threads：用来控制入队tensors线程的数量，如果num_threads大于1，则batch操作将是非确定性的，输出的batch可能会乱序
    capacity：一个整数，用来设置队列中元素的最大数量
    enqueue_many：在tensors中的tensor是否是单个样本
    shapes：可选，每个样本的shape，默认是tensors的shape
    dynamic_pad：Boolean值.允许输入变量的shape，出队后会自动填补维度，来保持与batch内的shapes相同
    allow_samller_final_batch：可选，Boolean值，如果为True队列中的样本数量小于batch_size时，出队的数量会以最终遗留下来的样本进行出队，如果为Flalse，小于batch_size的样本不会做出队处理
    shared_name：可选，通过设置该参数，可以对多个会话共享队列
    name：可选，操作的名字
'''

with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer()) 
    tf.local_variables_initializer().run() 
    #创建一个协调器，管理线程
    coord = tf.train.Coordinator()  

    #启动QueueRunner, 此时文件名才开始进队。
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)  
   
    try:
        while not coord.should_stop():
            
            img1,img_name1,img_height1,img_width1,gtboxes_and_label1,num_objects1 = sess.run([img,img_name,img_height,img_width,gtboxes_and_label,num_objects])  
            for i in range(batch_size):

                print(img_name1,img_height1,img_width1,gtboxes_and_label1,num_objects1)
            
    #终止线程
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)