import os
import time

import numpy as np
import tensorflow as tf

from frameLoader import Frameloader
from models.research.slim.nets import resnet_v1
from preprocessing.cropper import Cropper

slim = tf.contrib.slim

framePath = ["/home/boy2/UCF101/ucf101_dataset/frames/jpegs_256", "/home/boy2/UCF101/ucf101_dataset/flows/tvl1_flow/u",
             "/home/boy2/UCF101/ucf101_dataset/flows/tvl1_flow/v"]
featureStorePath = ["/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_crop_v3",
                    "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop_v3/u",
                    "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop_v3/v"]
ws_image = ["/home/boy2/UCF101/ucf101_dataset/features/frame_ws_3"]
width = 224
height = 224
color_channels = 3


def gen_frame_feature_resNet152(fpath, spath):
    """
    Read all video frames from framePath, then generate feature for each of them by resNet152. Store the features
    in featureStorePath.
    :return:
    """
    fl = Frameloader(fpath)
    fl.validate(spath)
    # fl.frame_parent_paths = fl.shuffle(fl.frame_parent_paths)
    # reader = Npyfilereader(fpath)
    # name, content = reader.read_npys()

    # load the resNet152 mode and the pre-trained weights and bias.
    input_layer = tf.placeholder(dtype=tf.float32, shape=[None, width, height, color_channels])

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            resNet152, end_points = resnet_v1.resnet_v1_152(input_layer,
                                                            num_classes=None,
                                                            is_training=False,
                                                            global_pool=True,
                                                            output_stride=None,
                                                            spatial_squeeze=True,
                                                            reuse=tf.AUTO_REUSE,
                                                            scope='resnet_v1_152')
    saver = tf.train.Saver()
    with tf.device('/device:GPU:0'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, "/home/boy2/UCF101/src/resNet-152/resnet_v1_152.ckpt")
            saver.save(sess, "/home/boy2/UCF101/src/resNet-152/temp")
            # read all video frames and generate frame features by resNet152
            while len(fl.frame_parent_paths) != 0:
                start = time.time()
                video_name = fl.get_current_video_name()
                temp = fl.load_frames()
                print("Current working on: ", video_name)
                # load next available video frames

                # crop the frame to 224 * 224
                cropper = Cropper(np.array(temp), (width, height))
                frames = cropper.crop_flip()
                # frames = np.reshape(frames, (1, width, height, color_channels))
                # frames = np.array([cv2.resize(t, (224, 224), interpolation=cv2.INTER_CUBIC) for t in temp], dtype=np.float32)
                if frames != []:
                    # split video frames into batches with size 100
                    num_chunks = np.ceil(len(frames) / 100)
                    chunks = np.array_split(np.array(frames), num_chunks)
                    feature = []
                    for c in chunks:
                        # make input tensor for resNet152
                        feature += list(np.reshape(sess.run(resNet152, feed_dict={input_layer: c}),
                                                   newshape=[len(c), 2048]))
                        # ep = sess.run(end_points, feed_dict={input_layer: c})
                    feature = np.array(feature)
                    np.save(os.path.join(spath, video_name), feature)
                # feature = sess.run(resNet152, feed_dict={input_layer: frames})
                # feature = np.ndarray.flatten(feature)
                # np.save(os.path.join(spath, video_name), feature)
                print(time.time() - start)


if __name__ == '__main__':
    for i in range(10):
        for f, s in zip(framePath, featureStorePath):
            gen_frame_feature_resNet152(framePath[2], featureStorePath[2])
