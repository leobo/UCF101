import os
import shutil

import numpy as np
import tensorflow as tf
from sklearn import preprocessing

from npyFileReader import Npyfilereader
from trainTestSamplesGen import TrainTestSampleGen
from weighted_sum import calWeightedSum
from weighted_sum.videoDescriptorWeightedSum import Weightedsum
from ws_unbalanced_video_clips.classifier_for_unbalance_video_clips import classify

# from ws_unbalanced_video_clips.two_stream_classifier import classify

num_samples_per_training_video = 1
num_samples_per_testing_video = 1
train_steps = 10
test_steps = 5
dims = 2
min_len_video = 30
max_len_video = 1000
num_train_data = 9537
num_test_data = 3783


def norm_encode_data(train_data, test_data):
    len_train = len(train_data)
    # normalize the data in time direction
    temp = np.concatenate([train_data, test_data])
    if len(temp.shape) == 3:
        x, y, t = temp.shape
        temp_norm = np.zeros(shape=(x, y, t))
        for i in range(t):
            temp_norm[:, :, i] = preprocessing.normalize(temp[:, :, i], axis=0)
    else:
        temp_norm = preprocessing.normalize(temp, axis=0)
    # optional: normalize each sample data independently
    # temp_norm = preprocessing.normalize(temp_norm, axis=1)
    train_data = temp_norm[:len_train]
    test_data = temp_norm[len_train:]
    return train_data, test_data


# def reformat(data_label_image, data_label_flow_1, data_label_flow_2):
#     data = []
#     for i, f1, f2 in zip(data_label_image['data'], data_label_flow_1['data'], data_label_flow_2['data']):
#         temp = []
#         # for j in range(len(i[0])):
#         temp.append(i)
#         temp.append(f1)
#         temp.append(f2)
#         data.append(temp)
#     return {'data': np.array(data), 'label': data_label_image['label']}
#
#
# def reformat_reshaped(data_label_image, data_label_flow_1, data_label_flow_2):
#     data = []
#     for i, f1, f2 in zip(data_label_image['data'], data_label_flow_1['data'], data_label_flow_2['data']):
#         temp = []
#         # for j in range(len(i[0])):
#         temp.append(np.reshape(i, newshape=(2048 * 2, 1)))
#         temp.append(np.reshape(f1, newshape=(2048 * 2, 1)))
#         temp.append(np.reshape(f2, newshape=(2048 * 2, 1)))
#         data.append(temp)
#     return {'data': np.array(data), 'label': data_label_image['label']}
#
#
# def reformat_flow(data_label_flow_1, data_label_flow_2):
#     data = []
#     for f1, f2 in zip(data_label_flow_1['data'], data_label_flow_2['data']):
#         temp = []
#         temp.append(f1)
#         temp.append(f2)
#         data.append(temp)
#     return {'data': np.array(data), 'label': data_label_flow_1['label']}
#
#
# def ws_flows(flow1_path, flow2_path, save_path1, save_path2, dim):
#     # flow 1
#     nr1 = Npyfilereader(flow1_path)
#     nr1.validate(save_path1)
#     # flow 2
#     nr2 = Npyfilereader(flow2_path)
#     nr2.validate(save_path2)
#
#     video_num = len(nr1.npy_paths)
#     for i in range(video_num):
#         name1, contents1 = nr1.read_npys()
#         name2, contents2 = nr2.read_npys()
#         ws1 = Weightedsum(name1, contents1, save_path1)
#         ws2 = Weightedsum(name2, contents2, save_path2)
#         if dim == 0:
#             ws1.mean_descriptor_gen()
#             ws2.mean_descriptor_gen()
#         else:
#             trans_m = ws1.transformation_matrix_gen(dim, ws1.frame_features.shape[0])
#             ws1.ws_descriptor_gen(dim, trans_matrix=trans_m)
#             ws2.ws_descriptor_gen(dim, trans_matrix=trans_m)
#
#
# def check_ws_existence(resNet_feature_save_path, resNet_ws_save_path, dim, flip=False):
#     feature = []
#     for (dirpath, dirnames, filenames) in os.walk(resNet_ws_save_path):
#         feature += [f for f in filenames if f.endswith('.npy')]
#     if len(feature) == 0:
#         calWeightedSum.calculate_weightedsum(resNet_feature_save_path, resNet_ws_save_path, dim, flip)
#
#
# def train_test_split_encode(tts, resNet_ws_save_path, dataset, split_num, encoder):
#     # resNet image feature
#     train_data_label_image, test_data_label_image = tts.train_test_split(resNet_ws_save_path, dataset, split_num)
#     # normalize the data and encode labels
#     train_data_label_image, test_data_label_image = norm_encode_data(train_data_label_image, test_data_label_image,
#                                                                      encoder)
#     return train_data_label_image, test_data_label_image
#
#
# def remove_dirctories(directories):
#     for d in directories:
#         shutil.rmtree(d)
#         os.makedirs(d)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def tf_record_writer(path, name, rgb, flow, label):
    if not os.path.exists(path):
        os.mkdir(path)
    writer = tf.python_io.TFRecordWriter(os.path.join(path, name))
    for _rgb, _flow, _label in zip(rgb, flow, label):

        feature = {'rgb': _bytes_feature(_rgb.astype(np.float32).tobytes()),
                   'flow': _bytes_feature(_flow.astype(np.float32).tobytes()),
                   'labels': _int64_feature(_label)
                   }
        # print(feature)
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()


def tf_record_writer_eval(path, name, rgb1, flow1, label1, rgb2, flow2, label2):
    if not os.path.exists(path):
        os.mkdir(path)
    writer = tf.python_io.TFRecordWriter(os.path.join(path, name))
    for _rgb1, _flow1, _label1, _rgb2, _flow2, _label2 in zip(rgb1, flow1, label1, rgb2, flow2, label2):
        feature = {'rgb': _bytes_feature(_rgb1.astype(np.float32).tobytes()),
                   'flow': _bytes_feature(_flow1.astype(np.float32).tobytes()),
                   'labels': _int64_feature(_label1)
                   }
        # print(feature)
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

        feature = {'rgb': _bytes_feature(_rgb2.astype(np.float32).tobytes()),
                   'flow': _bytes_feature(_flow2.astype(np.float32).tobytes()),
                   'labels': _int64_feature(_label2)
                   }
        # print(feature)
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()


def input_gen_all(rgb_feature_path, u_feature_path, v_feature_path, video_names, video_labels, steps=5, dim=2,
                  train=False):
    rgb_path = [os.path.join(rgb_feature_path, n + '.npy') for n in video_names][:]
    u_path = [os.path.join(u_feature_path, n + '.npy') for n in video_names][:]
    v_path = [os.path.join(v_feature_path, n + '.npy') for n in video_names][:]
    all_rgb = []
    all_u = []
    all_v = []
    all_label = []
    for rp, up, vp, name, l in zip(rgb_path, u_path, v_path, video_names[:],
                                   video_labels[:]):
        rgb = np.load(rp)
        u = np.load(up)
        v = np.load(vp)
        if train is False:
            selected_frame_nums = np.random.randint(steps, len(rgb) - steps, size=num_samples_per_testing_video,
                                                    dtype=np.int32)
        else:
            selected_frame_nums = np.random.randint(steps, len(rgb) - steps, size=num_samples_per_training_video,
                                                    dtype=np.int32)
        for index in selected_frame_nums:
            _rgb = Weightedsum(name, rgb[:], None).ws_descriptor_gen(dim, False, None)
            # _rgb = rgb[index]
            _u = Weightedsum(name, u[:], None).ws_descriptor_gen(dim, False, None)
            _v = Weightedsum(name, v[:], None).ws_descriptor_gen(dim, False, None)
            all_rgb.append(_rgb)
            all_u.append(_u)
            all_v.append(_v)
            all_label.append(l)
    return all_rgb, all_u, all_v, all_label


def input_gen(rgb_feature_path, u_feature_path, v_feature_path, video_names, video_labels, ws_feature_path, train=True,
              steps=5, dim=2):
    if not os.path.exists(ws_feature_path):
        # remove_dirctories(ws_feature_path)
        os.mkdir(ws_feature_path)

    writer = tf.python_io.TFRecordWriter(os.path.join(ws_feature_path, 'rgb_flow_labels.tfrecord'))
    rgb_path = [os.path.join(rgb_feature_path, n + '.npy') for n in video_names][:]
    u_path = [os.path.join(u_feature_path, n + '.npy') for n in video_names][:]
    v_path = [os.path.join(v_feature_path, n + '.npy') for n in video_names][:]

    for rp, up, vp, name, l in zip(rgb_path, u_path, v_path, video_names[:], video_labels[:]):
        rgb = np.load(rp)
        u = np.load(up)
        v = np.load(vp)
        if train is False:
            selected_frame_nums = np.random.randint(steps, len(rgb) - steps, size=num_samples_per_testing_video,
                                                    dtype=np.int32)
        else:
            selected_frame_nums = np.random.randint(steps, len(rgb) - steps, size=num_samples_per_training_video,
                                                    dtype=np.int32)
        for index in selected_frame_nums:
            if steps == 0:
                _rgb = rgb[index]
                _flow = []
                for i in range(len(u)):
                    _flow.append(u[i])
                    _flow.append(v[i])
            else:
                _rgb = rgb[index - steps: index + steps]
                _flow = []
                for i in range(index - steps, index + steps):
                    _flow.append(u[i])
                    _flow.append(v[i])
            # _rgb = np.transpose(np.reshape(rgb[index], newshape=(1, len(rgb[index]), 1)))
            # _rgb = Weightedsum(name, _rgb, None).ws_descriptor_gen(dim, False, None)
            # _u = u[index - steps: index + steps]
            # _v = v[index - steps: index + steps]
            _flow = Weightedsum(name, np.array(_flow), None).ws_descriptor_gen(dim, False, None)
            # _u = Weightedsum(name, _u, None).ws_descriptor_gen(dim, False, None)
            # _v = Weightedsum(name, _v, None).ws_descriptor_gen(dim, False, None)
            # t = preprocessing.normalize(np.concatenate((_rgb, _flow), axis=0), axis=1)
            # _rgb = t[:2048]
            # _flow = t[2048:]
            _rgb = preprocessing.normalize(np.reshape(_rgb, newshape=(2048, 1)), axis=0)
            _flow = preprocessing.normalize(_flow, axis=0)

            feature = {'rgb': _bytes_feature(_rgb.astype(np.float32).tobytes()),
                       'flow': _bytes_feature(_flow.astype(np.float32).tobytes()),
                       'labels': _int64_feature(l)
                       }
            # print(feature)
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
    writer.close()
    return len(rgb_path)


def crop_main(resNet_crop_save_path_v1, resNet_flow_crop_save_path_1_v1, resNet_flow_crop_save_path_2_v1,
              resNet_train_ws_v1, resNet_test_ws_v1,
              resNet_crop_save_path_v2, resNet_flow_crop_save_path_1_v2, resNet_flow_crop_save_path_2_v2,
              resNet_train_ws_v2, resNet_test_ws_v2,
              train_test_splits_save_path, dims, dataset='ucf'):
    if dataset == 'hmdb':
        tts = TrainTestSampleGen(ucf_path='', hmdb_path=train_test_splits_save_path)
    else:
        tts = TrainTestSampleGen(ucf_path=train_test_splits_save_path, hmdb_path='')

    accuracy = 0
    encoder = preprocessing.LabelEncoder()
    num_train = 0
    num_test = 0
    for i in range(1):
        # num_train += input_gen(resNet_crop_save_path_v1, resNet_flow_crop_save_path_1_v1,
        #                        resNet_flow_crop_save_path_2_v1,
        #                        tts.ucf_train_data_label[i]['data'],
        #                        encoder.fit_transform(tts.ucf_train_data_label[i]['label']),
        #                        resNet_train_ws_v1,
        #                        train=True, steps=train_steps, dim=dims)
        # num_test += input_gen(resNet_crop_save_path_v1, resNet_flow_crop_save_path_1_v1,
        #                       resNet_flow_crop_save_path_2_v1,
        #                       tts.ucf_test_data_label[i]['data'],
        #                       encoder.fit_transform(tts.ucf_test_data_label[i]['label']),
        #                       resNet_test_ws_v1,
        #                       train=False, steps=test_steps, dim=dims)

        # num_train += input_gen(resNet_crop_save_path_v2, resNet_flow_crop_save_path_1_v2,
        #                        resNet_flow_crop_save_path_2_v2,
        #                        tts.ucf_train_data_label[i]['data'],
        #                        encoder.fit_transform(tts.ucf_train_data_label[i]['label']),
        #                        resNet_train_ws_v1,
        #                        train=True, steps=train_steps, dim=dims)
        # num_test += input_gen(resNet_crop_save_path_v2, resNet_flow_crop_save_path_1_v2,
        #                       resNet_flow_crop_save_path_2_v2,
        #                       tts.ucf_test_data_label[i]['data'],
        #                       encoder.fit_transform(tts.ucf_test_data_label[i]['label']),
        #                       resNet_test_ws_v1,
        #                       train=False, steps=test_steps, dim=dims)

        # global num_train_data
        # num_train_data = num_train
        # global num_test_data
        # num_test_data = num_test
        # print("# of train videos is:", num_train)
        # print("# of test videos is:", num_test)

        # t_rgb, t_u, t_v, t_label = input_gen_all(resNet_crop_save_path_v1, resNet_flow_crop_save_path_1_v1,
        #                                          resNet_flow_crop_save_path_2_v1,
        #                                          tts.ucf_train_data_label[i]['data'],
        #                                          encoder.fit_transform(tts.ucf_train_data_label[i]['label']),
        #                                          steps=train_steps, dim=dims)
        # e_rgb, e_u, e_v, e_label = input_gen_all(resNet_crop_save_path_v1, resNet_flow_crop_save_path_1_v1,
        #                                          resNet_flow_crop_save_path_2_v1,
        #                                          tts.ucf_test_data_label[i]['data'],
        #                                          encoder.fit_transform(tts.ucf_test_data_label[i]['label']),
        #                                          steps=train_steps, dim=dims)
        # t_rgb, e_rgb = norm_encode_data(t_rgb, e_rgb)
        # t_u, e_u = norm_encode_data(t_u, e_u)
        # t_v, e_v = norm_encode_data(t_v, e_v)
        # #
        # _t_rgb, _t_u, _t_v, _t_label = input_gen_all(resNet_crop_save_path_v2, resNet_flow_crop_save_path_1_v2,
        #                                              resNet_flow_crop_save_path_2_v2,
        #                                              tts.ucf_train_data_label[i]['data'],
        #                                              encoder.fit_transform(tts.ucf_train_data_label[i]['label']),
        #                                              steps=train_steps, train=True, dim=dims)
        # _e_rgb, _e_u, _e_v, _e_label = input_gen_all(resNet_crop_save_path_v2, resNet_flow_crop_save_path_1_v2,
        #                                              resNet_flow_crop_save_path_2_v2,
        #                                              tts.ucf_test_data_label[i]['data'],
        #                                              encoder.fit_transform(tts.ucf_test_data_label[i]['label']),
        #                                              steps=test_steps, train=False, dim=dims)
        # _t_rgb, _e_rgb = norm_encode_data(_t_rgb, _e_rgb)
        # _t_u, _e_u = norm_encode_data(_t_u, _e_u)
        # _t_v, _e_v = norm_encode_data(_t_v, _e_v)
        #
        # # tf_record_writer(resNet_train_ws_v2, "rgb_flow_labels.tfrecord", _t_rgb,
        # #                  np.stack([_t_u, _t_v], axis=1), _t_label)
        # # tf_record_writer(resNet_test_ws_v2, "rgb_flow_labels.tfrecord", _e_rgb,
        # #                  np.stack([_e_u, _e_v], axis=1), _e_label)
        #
        # tf_record_writer(resNet_train_ws_v1, "rgb_flow_labels.tfrecord", np.concatenate((t_rgb, _t_rgb), axis=0),
        #                  np.stack([np.concatenate((t_u, _t_u), axis=0), np.concatenate((t_v, _t_v), axis=0)], axis=1),
        #                  np.concatenate((t_label, _t_label), axis=0))
        # tf_record_writer_eval(resNet_test_ws_v1, "rgb_flow_labels.tfrecord", e_rgb, np.stack([e_u, e_v], axis=1),
        #                       e_label, _e_rgb, np.stack([_e_u, _e_v], axis=1), _e_label)

        accuracy += classify(os.path.join(resNet_train_ws_v2, "rgb_flow_labels.tfrecord"),
                             os.path.join(resNet_test_ws_v2, "rgb_flow_labels.tfrecord"),
                             2*num_train_data * num_samples_per_training_video,
                             2*num_test_data * num_samples_per_testing_video,
                             2*num_samples_per_testing_video)
        print("accuracy is", accuracy)


if __name__ == '__main__':
    ucf_resNet_flow_crop_save_path_1_v3 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop_v3/u"
    ucf_resNet_flow_crop_save_path_2_v3 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop_v3/v"
    ucf_resNet_crop_save_path_v3 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_crop_v3"

    ucf_resNet_flow_crop_save_path_1_v2 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop_v2/u"
    ucf_resNet_flow_crop_save_path_2_v2 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop_v2/v"
    ucf_resNet_crop_save_path_v2 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_crop_v2"

    ucf_resNet_flow_crop_save_path_1_v1 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop/u"
    ucf_resNet_flow_crop_save_path_2_v1 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop/v"
    ucf_resNet_crop_save_path_v1 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_crop"

    ucf_resNet_flow_save_path_1 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow/u"
    ucf_resNet_flow_save_path_2 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow/v"
    ucf_resNet_save_path = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet"

    ucf_resNet_train_path_v1 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_train_v1"
    ucf_resNet_test_path_v1 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_test_v1"

    ucf_resNet_train_path_v2 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_train_v2"
    ucf_resNet_test_path_v2 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_test_v2"

    ucf_resNet_train_path_v3 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_train_v3"
    ucf_resNet_test_path_v3 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_test_v3"

    ucf_train_test_splits_save_path = "/home/boy2/UCF101/ucf101_dataset/features/testTrainSplits"

    # remove_dirctories([ucf_resNet_crop_ws_save_path_v1, ucf_resNet_crop_flow_ws_save_path_1_v1,
    #                    ucf_resNet_crop_flow_ws_save_path_2_v1,
    #                    ucf_resNet_crop_ws_save_path_v2, ucf_resNet_crop_flow_ws_save_path_1_v2,
    #                    ucf_resNet_crop_flow_ws_save_path_2_v2])

    # main(ucf_resNet_save_path, ucf_resNet_ws_save_path, ucf_resNet_flip_ws_save_path,
    #      ucf_resNet_crop_save_path, ucf_resNet_crop_ws_save_path, ucf_resNet_crop_flip_ws_save_path,
    #      ucf_resNet_flow_save_path_1, ucf_resNet_flow_ws_save_path_1, ucf_resNet_flip_flow_ws_save_path_1,
    #      ucf_resNet_flow_crop_save_path_1, ucf_resNet_crop_flow_ws_save_path_1,
    #      ucf_resNet_crop_flip_flow_ws_save_path_1,
    #      ucf_resNet_flow_save_path_2, ucf_resNet_flow_ws_save_path_2, ucf_resNet_flip_flow_ws_save_path_2,
    #      ucf_resNet_flow_crop_save_path_2, ucf_resNet_crop_flow_ws_save_path_2,
    #      ucf_resNet_crop_flip_flow_ws_save_path_2,
    #      ucf_train_test_splits_save_path, 2)

    crop_main(ucf_resNet_crop_save_path_v1, ucf_resNet_flow_crop_save_path_1_v1, ucf_resNet_flow_crop_save_path_2_v1,
              ucf_resNet_train_path_v1, ucf_resNet_test_path_v1,
              ucf_resNet_crop_save_path_v3, ucf_resNet_flow_crop_save_path_1_v3, ucf_resNet_flow_crop_save_path_2_v3,
              ucf_resNet_train_path_v3, ucf_resNet_test_path_v3,
              ucf_train_test_splits_save_path, dims)

# if __name__ == '__main__':
#     a = np.array([1,2,3]).reshape((1,3))
#     b = np.array([[1,2,3], [4,5,6]])
#     _a = preprocessing.normalize(a, axis=1)
#     _b = preprocessing.normalize(b, axis=1)
#     print(_a)
#     print(_b)
