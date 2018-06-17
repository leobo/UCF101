import os

import cv2
import numpy as np
from scipy.stats import ortho_group


class Weightedsum(object):

    def __init__(self, name, features, store_path):
        self.frame_features = features
        self.name = name
        self.store_path = store_path

    def matrix_multiply(self, m1, m2):
        """
        Calculate the multiplication of matrices m1 (n*m) and m2(m*k)
        :param matrix1: the first matrix
        :param matrix2: the second matrix
        :return: the multiplication of

         m1 and m2
        """
        return np.matmul(m1, m2)

    def transformation_matrix_gen(self, r, c):
        """
        Generate the transformation matrix. The transformation matrix has dimension r*c and r<=c. Also, the transformation
        matrix contain r linear independent column vectors.
        :param r: the row number
        :param c: the column number
        :return: the transformation matrix
        """
        if r == 2:
            temp = np.vstack((np.ones(shape=(c)), np.arange(1, c+1)))
            # _, r = np.linalg.qr(temp, mode='complete')
            # temp = np.vstack((np.ones(shape=c), np.arange(-c/2, c/2)))
            return temp

        rand_matrix = np.random.rand(r, c)
        _, r = np.linalg.qr(rand_matrix, mode='complete')
        return r

    def ws_descriptor_gen(self, r, save=True, trans_matrix=None):
        """
        Generate the video descriptor on top of frame descriptor (global or local) f_des by mapping certain elements in
        every frame features along time axis onto the plane which is described by trans_matrix.
        :param f_des: The frame features for a videos.
        :param r: The rank of the transformation matrix.
        :return: The video descriptor.
        """
        if trans_matrix is None:
            trans_matrix = self.transformation_matrix_gen(r, self.frame_features.shape[0])
        temp = np.transpose(self.matrix_multiply(trans_matrix, self.frame_features))
        # temp = self.post_processing(temp)
        if save:
            self.save_des(temp)
        return temp

    def ws_on_raw_data(self, r):
        trans_matrix = self.transformation_matrix_gen(r, self.frame_features.shape[0])
        temp = np.swapaxes(self.frame_features, 0, 2)
        temp = np.tensordot(temp, trans_matrix, axes=((2), (1)))
        self.save_des(temp)
        return temp

    def mean_descriptor_gen(self):
        m = np.mean(self.frame_features, axis=0)
        self.save_des(m)

    def post_processing(self, temp):
        post_processed_feature = []
        for frame_feature in temp:
            for line in frame_feature:
                post_processed_feature += [line]
        return np.array(post_processed_feature)

    def pre_processing(self):
        pre_processed_feature = []
        i = 0
        for frame_feature in self.frame_features:
            if frame_feature is None:
                return -1
            for line in frame_feature:
                pre_processed_feature += [line]
            i = i + 1
        self.frame_features = np.array(pre_processed_feature)

    def ws_descriptor_post_process(self, v_des):
        """
        For test.....
        :param v_des:
        :return:
        """
        post_pro_ws_des = []
        for v in v_des:
            temp = []
            for feature in v:
                temp = np.concatenate((temp, feature), axis=0)
            post_pro_ws_des.append(temp)
        return post_pro_ws_des

    def save_des(self, descriptors):
        """
        Save the descriptors under self.video_store_path/self.video_name with .npy format
        :param descriptors: the given all frames descriptors of one video
        """
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)
        np.save(os.path.join(self.store_path, self.name), descriptors)

    def save_as_image(self, data):
        cv2.imwrite(os.path.join(self.store_path, self.name + '.jpg'), data)
