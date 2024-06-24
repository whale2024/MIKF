"""
 -*- coding:utf-8 -*-
 @FileName: MIKF.py
 @Author: zjy
 @DateTime: 2024/6/24 20:59
 @Description:
 @IDE:PyCharm
"""
import os
import warnings
from time import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import euclidean_distances, accuracy_score, f1_score, zero_one_loss
from sklearn.svm import SVC

from MILFrame.I2I import kernel_rbf
from MILFrame.MILData import MILData

warnings.filterwarnings('ignore')


def norm(x):
    # 计算每个元素的指数
    exps = np.exp(x)
    # print(x)
    # 计算每个元素的指数和
    exps_sum = np.sum(exps)
    # 计算softmax函数的值
    softmax = exps / exps_sum
    return softmax


class MIKF:
    def __init__(self, mil_data: MILData, tau=10):
        self.E = None
        self.data = mil_data
        self.tau = tau

    def __esk(self, embedding_space: np.ndarray):
        vec = []
        es_shape = embedding_space.shape[0]
        for idx in range(self.data.num_bag):
            bag = self.data.get_bag(idx)
            vec.append(euclidean_distances(bag, embedding_space).mean(0).reshape(1, es_shape))
        S = np.zeros((self.data.num_bag, self.data.num_bag))
        for i in range(self.data.num_bag):
            for j in range(self.data.num_bag):
                S[i][j] = kernel_rbf(vec[i], vec[j])

        return S

    def __embedding_space_construction(self, tr_idx):

        po_lab, ne_lab = np.max(self.data.bag_labels), np.min(self.data.bag_labels)

        tr_lab = self.data.bag_labels[tr_idx]

        po_idx, ne_idx = np.where(tr_lab == po_lab)[0], np.where(tr_lab == ne_lab)[0]

        ne_ins = self.data.get_instance_space(ne_idx)

        phi = len(po_idx)

        k_means = MiniBatchKMeans(n_clusters=phi)
        k_means.fit(ne_ins)
        M_1 = k_means.cluster_centers_
        NI_lab = np.zeros(len(M_1))
        ins_lab = np.concatenate((NI_lab, [1]))
        PI = []
        BI = []
        for j in range(phi):

            pos_bag = self.data.get_bag(po_idx[j])
            svm = SVC(kernel="linear", C=100000)

            score = []

            # euclidean-based score
            # score=euclidean_distances(pos_bag, M_1).sum(1)

            for i in range(len(pos_bag)):
                ins = pos_bag[i]
                vec = list(M_1)
                vec.append(ins)
                svm.fit(vec, ins_lab)
                score.append(2 / np.linalg.norm(svm.coef_))

            alpha_idx = np.argmax(score)
            beta_idx = np.argmin(score)
            PI.append(pos_bag[alpha_idx].tolist())
            BI.append(pos_bag[beta_idx].tolist())

        M_2 = np.array(PI)
        M_3 = np.array(BI)

        return M_1, M_2, M_3

    def __get_E_matrix(self, tr_idx):
        pos_bag_idx = []
        neg_bag_idx = []

        pos_label = self.data.pos_label
        for i in tr_idx:
            if self.data.bag_labels[i] == pos_label:
                pos_bag_idx.append(i)
            else:
                neg_bag_idx.append(i)
        num_pos = len(pos_bag_idx)
        num_neg = len(neg_bag_idx)
        Z_1 = (1 / (num_pos ** 2 + num_neg ** 2))
        Z_2 = -1 / (num_pos * num_neg * 2)

        E_matrix = np.zeros((self.data.num_bag, self.data.num_bag))
        for i in range(self.data.num_bag):
            for j in range(self.data.num_bag):
                if self.data.bag_labels[tr_idx[i]] == self.data.bag_labels[tr_idx[j]]:
                    E_matrix[i][j] = Z_1
                else:
                    E_matrix[i][j] = Z_2

        self.E = E_matrix

    def H(self, S_m, tr_idx):
        S_m = S_m[:, tr_idx]
        S_m = S_m[tr_idx, :]

        num_bag = len(tr_idx)
        temp_scores = 0
        for i in range(num_bag):
            for j in range(num_bag):
                temp_scores += S_m[i][j] * self.E[i][j]
        h_m = temp_scores / 2
        return h_m

    def __kernel_fusion(self, tr_idx):

        KERNEL_MATRIX = np.zeros((self.data.num_bag, self.data.num_bag))

        M_1, M_2, M_3 = self.__embedding_space_construction(tr_idx)
        S_1, S_2, S_3 = self.__esk(M_1), self.__esk(M_2), self.__esk(M_3)
        h_1, h_2, h_3 = self.H(S_1, tr_idx), self.H(S_2, tr_idx), self.H(S_3, tr_idx)

        h = np.array([h_1, h_2, h_3])
        w = norm(h)

        for i in range(self.data.num_bag):
            for j in range(self.data.num_bag):
                KERNEL_MATRIX[i][j] = self.tau * (w[0] * S_1[i][j] + w[1] * S_2[i][j] + w[2] * S_3[i][j])

        return KERNEL_MATRIX

    def main(self, k):
        """"""
        tr_idxes, te_idxes = self.data.get_k_cv_idx(k)

        classifier = SVC(kernel="precomputed", C=1)

        HAT_Y, Y = [], []

        for i, (tr_idx, te_idx) in enumerate(zip(tr_idxes, te_idxes)):
            # print("{}-th fold.".format(i))

            KERNEL_MATRIX = self.__kernel_fusion(tr_idx)[:, tr_idx]

            tr_KERNEL, te_KERNEL = KERNEL_MATRIX[tr_idx], KERNEL_MATRIX[te_idx]

            classifier.fit(tr_KERNEL, self.data.bag_labels[tr_idx])

            hat_te_y = classifier.predict(te_KERNEL)

            HAT_Y.extend(hat_te_y.tolist())

            Y.extend(self.data.bag_labels[te_idx].tolist())

        return accuracy_score(Y, HAT_Y), f1_score(Y, HAT_Y), zero_one_loss(Y, HAT_Y)
