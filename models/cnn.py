#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

from models.module import Conv_Module, Encoder, Decoder
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
import cupy
import numpy
from chainer import Variable, cuda

from logging import getLogger
logger = getLogger("main")


class CNNBase(chainer.Chain):
    def __init__(self, mean, std, gpu):
        super(CNNBase, self).__init__()
        self._mean = mean
        self._std = std
        self.nb_inputs = len(mean)
        self.target_idx = -1

        # Send mean and sd of the dataset to GPU to produce prdiction result in the image coordinate
        self.mean = Variable(cuda.to_gpu(mean.astype(np.float32), gpu))
        self.std = Variable(cuda.to_gpu(std.astype(np.float32), gpu))

    def _prepare_input(self, inputs):
        pos_x, pos_y, poses, egomotions = inputs[:4]
        if pos_y.data.ndim == 2:
            pos_x = F.expand_dims(pos_x, 0)
            pos_y = F.expand_dims(pos_y, 0)
            if egomotions is not None:
                egomotions = F.expand_dims(egomotions, 0)
            poses = F.expand_dims(poses, 0)

        # Locations
        # Note: prediction target is displacement from last input
        x = (pos_x - F.broadcast_to(self.mean, pos_x.shape)) / \
            F.broadcast_to(self.std, pos_x.shape)
        y = (pos_y - F.broadcast_to(self.mean, pos_y.shape)) / \
            F.broadcast_to(self.std, pos_y.shape)
        y = y - F.broadcast_to(x[:, -1:, :], pos_y.shape)

        # Egomotions
        past_len = pos_x.shape[1]
        if egomotions is not None:
            ego_x = egomotions[:, :past_len, :]
            ego_y = egomotions[:, past_len:, :]

        # Poses
        poses = F.reshape(poses, (poses.shape[0], poses.shape[1], -1))
        pose_x = poses[:, :past_len, :]
        pose_y = poses[:, past_len:, :]

        if egomotions is not None:
            return x, y, x[:, -1, :], ego_x, ego_y, pose_x, pose_y
        else:
            return x, y, x[:, -1, :], None, None, pose_x, pose_y

    def predict(self, inputs):
        return self.__call__(inputs)


class CNN(CNNBase):
    """
    Baseline: location only
    """

    def __init__(self, mean, std, gpu, channel_list, dc_channel_list, ksize_list,
                 dc_ksize_list, inter_list, last_list, pad_list):
        super(CNN, self).__init__(mean, std, gpu)
        if len(ksize_list) > 0 and len(dc_ksize_list) == 0:
            dc_ksize_list = ksize_list
        with self.init_scope():
            self.pos_encoder = Encoder(
                self.nb_inputs, channel_list, ksize_list, pad_list)
            self.pos_decoder = Decoder(
                dc_channel_list[-1], dc_channel_list, dc_ksize_list[::-1])
            self.inter = Conv_Module(
                channel_list[-1], dc_channel_list[0], inter_list)
            self.last = Conv_Module(
                dc_channel_list[-1], self.nb_inputs, last_list, True)

    def __call__(self, inputs):
        pos_x, pos_y, offset_x, ego_x, ego_y, pose_x, pose_y = self._prepare_input(
            inputs)
        batch_size, past_len, _ = pos_x.shape

        h = self.pos_encoder(pos_x)
        h = self.inter(h)
        h = self.pos_decoder(h)
        pred_y = self.last(h)
        pred_y = F.swapaxes(pred_y, 1, 2)
        pred_y = pred_y[:, :pos_y.shape[1], :]
        loss = F.mean_squared_error(pred_y, pos_y)

        pred_y = pred_y + \
            F.broadcast_to(F.expand_dims(offset_x, 1), pred_y.shape)
        pred_y = cuda.to_cpu(pred_y.data) * self._std + self._mean
        return loss, pred_y, None


class CNN_Ego(CNNBase):
    """
    Baseline: feeds locations and egomotions
    """

    def __init__(self, mean, std, gpu, channel_list, dc_channel_list, ksize_list,
                 dc_ksize_list, inter_list, last_list, pad_list, ego_type):
        super(CNN_Ego, self).__init__(mean, std, gpu)
        ego_dim = 6 if ego_type == "sfm" else 96 if ego_type == "grid" else 24
        if len(ksize_list) > 0 and len(dc_ksize_list) == 0:
            dc_ksize_list = ksize_list
        with self.init_scope():
            self.pos_encoder = Encoder(
                self.nb_inputs, channel_list, ksize_list, pad_list)
            self.ego_encoder = Encoder(
                ego_dim, channel_list, ksize_list, pad_list)
            self.pos_decoder = Decoder(
                dc_channel_list[-1], dc_channel_list, dc_ksize_list[::-1])
            self.inter = Conv_Module(
                channel_list[-1]*2, dc_channel_list[0], inter_list)
            self.last = Conv_Module(
                dc_channel_list[-1], self.nb_inputs, last_list, True)

    def __call__(self, inputs):
        pos_x, pos_y, offset_x, ego_x, ego_y, pose_x, pose_y = self._prepare_input(
            inputs)
        batch_size, past_len, _ = pos_x.shape

        h_pos = self.pos_encoder(pos_x)
        h_ego = self.ego_encoder(ego_x)
        h = F.concat((h_pos, h_ego), axis=1)  # (B, C, 2)
        h = self.inter(h)
        h_pos = self.pos_decoder(h)
        pred_y = self.last(h_pos)  # (B, 10, C+6+28)
        pred_y = F.swapaxes(pred_y, 1, 2)
        pred_y = pred_y[:, :pos_y.shape[1], :]
        loss = F.mean_squared_error(pred_y, pos_y)

        pred_y = pred_y + \
            F.broadcast_to(F.expand_dims(offset_x, 1), pred_y.shape)
        pred_y = cuda.to_cpu(pred_y.data) * self._std + self._mean
        return loss, pred_y, None


class CNN_Pose(CNNBase):
    """
    Baseline: feeds locations and poses
    """

    def __init__(self, mean, std, gpu, channel_list, dc_channel_list, ksize_list,
                 dc_ksize_list, inter_list, last_list, pad_list):
        super(CNN_Pose, self).__init__(mean, std, gpu)
        if len(ksize_list) > 0 and len(dc_ksize_list) == 0:
            dc_ksize_list = ksize_list
        with self.init_scope():
            self.pos_encoder = Encoder(
                self.nb_inputs, channel_list, ksize_list, pad_list)
            self.pose_encoder = Encoder(36, channel_list, ksize_list, pad_list)
            self.pos_decoder = Decoder(
                dc_channel_list[-1], dc_channel_list, dc_ksize_list[::-1])
            self.inter = Conv_Module(
                channel_list[-1]*2, dc_channel_list[0], inter_list)
            self.last = Conv_Module(
                dc_channel_list[-1], self.nb_inputs, last_list, True)

    def __call__(self, inputs):
        pos_x, pos_y, offset_x, ego_x, ego_y, pose_x, pose_y = self._prepare_input(
            inputs)
        batch_size, past_len, _ = pos_x.shape

        h_pos = self.pos_encoder(pos_x)
        h_pose = self.pose_encoder(pose_x)
        h = F.concat((h_pos, h_pose), axis=1)  # (B, C, 2)
        h = self.inter(h)
        h_pos = self.pos_decoder(h)
        pred_y = self.last(h_pos)  # (B, 10, C+6+28)
        pred_y = F.swapaxes(pred_y, 1, 2)
        pred_y = pred_y[:, :pos_y.shape[1], :]
        loss = F.mean_squared_error(pred_y, pos_y)

        pred_y = pred_y + \
            F.broadcast_to(F.expand_dims(offset_x, 1), pred_y.shape)
        pred_y = cuda.to_cpu(pred_y.data) * self._std + self._mean
        return loss, pred_y, None


class CNN_Ego_Pose(CNNBase):
    """
    Our full model: feeds locations, egomotions and poses as input
    """

    def __init__(self, mean, std, gpu, channel_list, dc_channel_list, ksize_list,
                 dc_ksize_list, inter_list, last_list, pad_list, ego_type):
        super(CNN_Ego_Pose, self).__init__(mean, std, gpu)
        ego_dim = 6 if ego_type == "sfm" else 96 if ego_type == "grid" else 24
        if len(ksize_list) > 0 and len(dc_ksize_list) == 0:
            dc_ksize_list = ksize_list
        with self.init_scope():
            self.pos_encoder = Encoder(
                self.nb_inputs, channel_list, ksize_list, pad_list)
            self.ego_encoder = Encoder(
                ego_dim, channel_list, ksize_list, pad_list)
            self.pose_encoder = Encoder(36, channel_list, ksize_list, pad_list)
            self.pos_decoder = Decoder(
                dc_channel_list[-1], dc_channel_list, dc_ksize_list[::-1])
            self.inter = Conv_Module(
                channel_list[-1]*3, dc_channel_list[0], inter_list)
            self.last = Conv_Module(
                dc_channel_list[-1], self.nb_inputs, last_list, True)

    def __call__(self, inputs):
        pos_x, pos_y, offset_x, ego_x, ego_y, pose_x, pose_y = self._prepare_input(
            inputs)
        batch_size, past_len, _ = pos_x.shape

        print("pos_x:", pos_x.shape)
        print("pose_x:", pose_x.shape)
        print("ego_x:", ego_x.shape)
        h_pos = self.pos_encoder(pos_x)
        h_pose = self.pose_encoder(pose_x)
        h_ego = self.ego_encoder(ego_x)
        print("h_pos:", h_pos.shape)
        print("h_pose:", h_pose.shape)
        print("h_ego:", h_ego.shape)
        h = F.concat((h_pos, h_pose, h_ego), axis=1)  # (B, C, 2)
        h = self.inter(h)
        h_pos = self.pos_decoder(h)
        pred_y = self.last(h_pos)  # (B, 10, C+6+28)
        pred_y = F.swapaxes(pred_y, 1, 2)
        pred_y = pred_y[:, :pos_y.shape[1], :]
        print(pred_y.shape)
        loss = F.mean_squared_error(pred_y, pos_y)

        pred_y = pred_y + \
            F.broadcast_to(F.expand_dims(offset_x, 1), pred_y.shape)
        pred_y = cuda.to_cpu(pred_y.data) * self._std + self._mean
        return loss, pred_y, None


class LSTMBase(chainer.Chain):
    def __init__(self, mean, std, gpu):
        super(LSTMBase, self).__init__()
        print("进入LSTMBase初始化")
        self._mean = mean
        self._std = std
        self.nb_inputs = len(mean)
        self.target_idx = -1

        # Send mean and sd of the dataset to GPU to produce prdiction result in the image coordinate
        self.mean = Variable(cuda.to_gpu(mean.astype(np.float32), gpu))
        self.std = Variable(cuda.to_gpu(std.astype(np.float32), gpu))

    def _prepare_input(self, inputs):
        pos_x, pos_y, poses, egomotions = inputs[:4]
        if pos_y.data.ndim == 2:
            pos_x = F.expand_dims(pos_x, 0)
            pos_y = F.expand_dims(pos_y, 0)
            if egomotions is not None:
                egomotions = F.expand_dims(egomotions, 0)
            poses = F.expand_dims(poses, 0)

        # Locations
        # Note: prediction target is displacement from last input
        x = (pos_x - F.broadcast_to(self.mean, pos_x.shape)) / \
            F.broadcast_to(self.std, pos_x.shape)
        y = (pos_y - F.broadcast_to(self.mean, pos_y.shape)) / \
            F.broadcast_to(self.std, pos_y.shape)
        y = y - F.broadcast_to(x[:, -1:, :], pos_y.shape)

        # Egomotions
        past_len = pos_x.shape[1]
        if egomotions is not None:
            ego_x = egomotions[:, :past_len, :]
            ego_y = egomotions[:, past_len:, :]

        # Poses
        poses = F.reshape(poses, (poses.shape[0], poses.shape[1], -1))
        pose_x = poses[:, :past_len, :]
        pose_y = poses[:, past_len:, :]

        if egomotions is not None:
            return x, y, x[:, -1, :], ego_x, ego_y, pose_x, pose_y
        else:
            return x, y, x[:, -1, :], None, None, pose_x, pose_y

    def predict(self, inputs):
        return self.__call__(inputs)


class LSTM_Encoder(chainer.Chain):
    def __init__(self, nb_inputs, nb_hidden, nb_outputs):
        super(LSTM_Encoder, self).__init__()
        with self.init_scope():
            self.lstm1 = L.NStepLSTM(
                n_layers=1, in_size=nb_inputs, out_size=nb_hidden, dropout=0.5)
            # self.lstm2 = L.LSTM(nb_hidden, nb_hidden)
            # self.lstm3 = L.LSTM(nb_hidden, nb_hidden)
            # self.fc = L.Linear(nb_hidden, nb_outputs)

    def reset_state(self):
        self.lstm1.reset_state()
        # self.lstm2.reset_state()
        # self.lstm3.reset_state()

    def __call__(self, hx, cx, xs):
        # h = F.swapaxes(x, 1, 2)  # (B, D, L)
        # print("输入lstm的数据的大小:", h.shape)
        # h = self.lstm1(h)
        # print("1层lstm输出的数据的大小:", h.shape)
        # h = self.lstm2(h)
        # h = self.lstm3(h)
        # 将输入数据转换成列表

        # 调用NStepLSTM层，得到输出数据ys
        hy, cy, ys = self.lstm1(hx, cx, xs)

        return ys


class LSTM_Decoder(chainer.Chain):
    def __init__(self, nb_inputs, nb_hidden, nb_outputs):
        super(LSTM_Encoder, self).__init__()
        with self.init_scope():
            self.lstm1 = L.NStepLSTM(
                n_layers=1, in_size=nb_inputs, out_size=nb_hidden, dropout=0.5)
            # self.lstm2 = L.LSTM(nb_hidden, nb_hidden)
            # self.lstm3 = L.LSTM(nb_hidden, nb_hidden)
            self.fc = L.Linear(nb_hidden, nb_outputs)

    def reset_state(self):
        self.lstm1.reset_state()
        # self.lstm2.reset_state()
        # self.lstm3.reset_state()

    def __call__(self, hx, cx, xs):
        # h = F.swapaxes(x, 1, 2)  # (B, D, L)
        # print("输入lstm的数据的大小:", h.shape)
        # h = self.lstm1(h)
        # print("1层lstm输出的数据的大小:", h.shape)
        # h = self.lstm2(h)
        # h = self.lstm3(h)
        # 将输入数据转换成列表

        # 调用NStepLSTM层，得到输出数据ys
        hy, cy, ys = self.lstm1(hx, cx, xs)

        return ys

def handlerData(data):
    data = numpy.asarray(data).tolist()
    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                data[i][j][k] = np.float32(cupy.asnumpy(data[i][j][k].data))
        data[i] = chainer.Variable(chainer.backends.cuda.to_gpu(np.asarray(data[i])))
    return data


def handlerDataDecoder(data):
    data = data.tolist()
    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                data[i][j][k] = np.float32(cupy.asnumpy(data[i][j][k].data))
        data[i] = chainer.Variable(chainer.backends.cuda.to_gpu(np.asarray(data[i])))
    return data

def handlerLast(data):
    for i in range(len(data)):
        data[i] = chainer.backends.cuda.to_gpu(data[i].data)
        # for j in range(len(data[i])):
        #     for k in range(len(data[i][j])):
        #         data[i][j][k] = np.float32(data[i][j][k].get())
    return chainer.Variable(cupy.asarray(data))

class LSTM_EGO_POS(LSTMBase):
    """
    Our full model: feeds locations, egomotions and poses as input
    """

    def __init__(self, mean, std, gpu, channel_list, dc_channel_list, ksize_list,
                 dc_ksize_list, inter_list, last_list, pad_list, ego_type):
        super(LSTM_EGO_POS, self).__init__(mean, std, gpu)
        print("进入LSTM模型")
        ego_dim = 6 if ego_type == "sfm" else 96 if ego_type == "grid" else 24
        if len(ksize_list) > 0 and len(dc_ksize_list) == 0:
            dc_ksize_list = ksize_list
        with self.init_scope():
            self.pos_encoder = LSTM_Encoder(3, 128, 10)
            self.ego_encoder = LSTM_Encoder(6, 128, 10)
            self.pose_encoder = LSTM_Encoder(36, 128, 10)
            self.pos_decoder = LSTM_Encoder(384, 128, 10)
            # self.inter = Conv_Module(
            #     channel_list[-1]*3, dc_channel_list[0], inter_list)
            print("dc_channel_list[-1]", dc_channel_list[-1])
            self.last = Conv_Module(128, self.nb_inputs, last_list, True)

    def __call__(self, inputs):
        pos_x, pos_y, offset_x, ego_x, ego_y, pose_x, pose_y = self._prepare_input(
            inputs)
        batch_size, past_len, _ = pos_x.shape

        # print("pos_x.type:", type(pos_x))
        # print("pos_x[0].type:", type(pos_x[0]))
        # print("pos_x[0][0] type:", type(pos_x[0][0]))
        # # print("pos_x[0][0][0]", pos_x[0][0][0])
        # print("pos_x[0][0][0] type", type(pos_x[0][0][0]))
        pos_x = handlerData(pos_x)
        pose_x = handlerData(pose_x)
        ego_x = handlerData(ego_x)

        # print("pos_x.type:", type(pos_x))
        # print("pos_x[0].type:", type(pos_x[0]))
        # print("pos_x[0][0] type:", type(pos_x[0][0]))
        # print("pos_x[0][0][0] type", type(pos_x[0][0][0]))
        h_pos = self.pos_encoder(None, None, pos_x)
        h_pose = self.pose_encoder(None, None, pose_x)
        h_ego = self.ego_encoder(None, None, ego_x)
        # print("h_pos:", h_pos.shape)
        # print("h_pose:", h_pose.shape)
        # print("h_ego:", h_ego.shape)

        h = np.concatenate((h_pos, h_pose, h_ego), axis=2)

        # h = F.concat((h_pos, h_pose, h_ego), axis=2)  # (B, C, 2)
        print("h.type:", type(h))
        print("h[0].type:", type(h[0]))
        print("h[0][0] type:", type(h[0][0]))
        print("h[0][0][0] type", type(h[0][0][0]))
        print("h:", h.shape)
        # h = self.inter(h)

        h = handlerDataDecoder(h)

        h_pos = self.pos_decoder(None, None, h)
        # print(h_pos)
        # print(type(h_pos))
        # print(h_pos.shape)

        print("h_pos.type:", type(h_pos))
        print("h_pos[0].type:", type(h_pos[0]))
        print("h_pos[0][0] type:", type(h_pos[0][0]))
        print("h_pos[0][0][0] type", type(h_pos[0][0][0]))
        # print("h_pos:", h_pos)

        # 进入last之前需要处理数据，适配self.last
        print("h_pos:", h_pos)
        h_pos = handlerLast(h_pos)


        # print("h_pos shape:",h_pos.shape)

        # print("h_pos.type:", type(h_pos))
        # print("h_pos[0].type:", type(h_pos[0]))
        # print("h_pos[0][0] type:", type(h_pos[0][0]))
        # print("h_pos[0][0][0] type", type(h_pos[0][0][0]))

        # (64, 10 ,128) -> (64, 128, 10)
        h_pos = F.swapaxes(h_pos, 1, 2)
        # print("h_pos shape:",h_pos.shape)

        pred_y = self.last(h_pos)  # (B, 10, C+6+28)
        pred_y = F.swapaxes(pred_y, 1, 2)
        pred_y = pred_y[:, :pos_y.shape[1], :]
        print(pred_y.shape)
        loss = F.mean_squared_error(pred_y, pos_y)

        pred_y = pred_y + \
            F.broadcast_to(F.expand_dims(offset_x, 1), pred_y.shape)
        pred_y = cuda.to_cpu(pred_y.data) * self._std + self._mean
        return loss, pred_y, None
