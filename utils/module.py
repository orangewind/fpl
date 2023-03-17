#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

from __future__ import print_function
from __future__ import division
from six.moves import range

import chainer
import chainer.functions as F
import chainer.links as L

from models.module import Conv_Module, Encoder, Decoder
import numpy as np

import cupy
from chainer import Variable, cuda


class LSTMBase(chainer.Chain):
    def __init__(self, mean, std, gpu):
        super(LSTMBase, self).__init__()
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


class Linear_BN(chainer.Chain):
    def __init__(self, nb_in, nb_out, no_bn=False):
        super(Linear_BN, self).__init__()
        self.no_bn = no_bn
        with self.init_scope():
            self.fc = L.Linear(nb_in, nb_out)
            if not no_bn:
                self.bn = L.BatchNormalization(nb_out)

    def __call__(self, x):
        if self.no_bn:
            return self.fc(x)
        else:
            return F.relu(self.bn(self.fc(x)))


class Conv_BN(chainer.Chain):
    def __init__(self, nb_in, nb_out, ksize=1, pad=0, no_bn=False):
        super(Conv_BN, self).__init__()
        self.no_bn = no_bn
        with self.init_scope():
            self.conv = L.ConvolutionND(1, nb_in, nb_out, ksize=ksize, pad=pad)
            if not no_bn:
                self.bn = L.BatchNormalization(nb_out)

    def __call__(self, x):
        if self.no_bn:
            return self.conv(x)
        else:
            return F.relu(self.bn(self.conv(x)))


class FC_Module(chainer.Chain):
    def __init__(self, nb_in, nb_out, inter_list=[], no_act_last=False):
        super(FC_Module, self).__init__()
        self.nb_layers = len(inter_list) + 1
        with self.init_scope():
            if len(inter_list) == 0:
                setattr(self, "fc1", Linear_BN(nb_in, nb_out, no_act_last))
            else:
                setattr(self, "fc1", Linear_BN(nb_in, inter_list[0]))
                for lidx, (nin, nout) in enumerate(zip(inter_list[:-1], inter_list[1:])):
                    setattr(self, "fc{}".format(lidx+2), Linear_BN(nin, nout))
                setattr(self, "fc{}".format(self.nb_layers),
                        Linear_BN(inter_list[-1], nb_out, no_act_last))

    def __call__(self, h, no_act_last=False):
        for idx in range(1, self.nb_layers + 1, 1):
            h = getattr(self, "fc{}".format(idx))(h)
        return h


class Conv_Module(chainer.Chain):
    def __init__(self, nb_in, nb_out, inter_list=[], no_act_last=False):
        super(Conv_Module, self).__init__()
        self.nb_layers = len(inter_list) + 1
        with self.init_scope():
            if len(inter_list) == 0:
                setattr(self, "layer1", Conv_BN(
                    nb_in, nb_out, no_bn=no_act_last))
            else:
                setattr(self, "layer1", Conv_BN(nb_in, inter_list[0]))
                for lidx, (nin, nout) in enumerate(zip(inter_list[:-1], inter_list[1:])):
                    setattr(self, "layer{}".format(lidx+2), Conv_BN(nin, nout))
                setattr(self, "layer{}".format(self.nb_layers), Conv_BN(
                    inter_list[-1], nb_out, no_bn=no_act_last))

    def __call__(self, h):
        for idx in range(1, self.nb_layers + 1, 1):
            h = getattr(self, "layer{}".format(idx))(h)
        return h


class Encoder(chainer.Chain):
    def __init__(self, nb_inputs, channel_list, ksize_list, pad=0):
        super(Encoder, self).__init__()
        self.nb_layers = len(channel_list)
        channel_list = [nb_inputs] + channel_list
        for idx, (nb_in, nb_out, ksize) in enumerate(zip(channel_list[:-1], channel_list[1:], ksize_list)):
            self.add_link("conv{}".format(idx), Conv_BN(
                nb_in, nb_out, ksize, pad))

    def __call__(self, x):
        h = F.swapaxes(x, 1, 2)  # (B, D, L)
        for idx in range(self.nb_layers):
            h = getattr(self, "conv{}".format(idx))(h)
        return h


class Decoder(chainer.Chain):
    def __init__(self, nb_inputs, channel_list, ksize_list, no_act_last=False):
        super(Decoder, self).__init__()
        self.nb_layers = len(channel_list)
        self.no_act_last = no_act_last
        channel_list = channel_list + [nb_inputs]
        for idx, (nb_in, nb_out, ksize) in enumerate(zip(channel_list[:-1], channel_list[1:], ksize_list[::-1])):
            self.add_link("deconv{}".format(idx),
                          L.DeconvolutionND(1, nb_in, nb_out, ksize))
            if no_act_last and idx == self.nb_layers - 1:
                continue
            self.add_link("bn{}".format(idx), L.BatchNormalization(nb_out))

    def __call__(self, h):
        for idx in range(self.nb_layers):
            if self.no_act_last and idx == self.nb_layers - 1:
                h = getattr(self, "deconv{}".format(idx))(h)
            else:
                h = F.relu(getattr(self, "bn{}".format(idx))(
                    getattr(self, "deconv{}".format(idx))(h)))
        return h
