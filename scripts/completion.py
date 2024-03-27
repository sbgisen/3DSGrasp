#!/usr/bin/env pipenv-shebang
# -*- coding:utf-8 -*-

# Copyright (c) 2023 SoftBank Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pathlib

import numpy as np
import rospkg
import rospy
import sensor_msgs.point_cloud2 as pc2
import torch
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header
from threeds_grasp.srv import CompleteCloud
from threeds_grasp.srv import CompleteCloudRequest
from threeds_grasp.srv import CompleteCloudResponse
from tools import builder
from tools import runner
from utils.config import get_config


class Completion(object):

    class Arg(object):

        def __init__(self) -> None:
            pkg = rospkg.RosPack().get_path('threeds_grasp')
            self.ckpts = rospy.get_param('~model', pkg + '/config/3dsgrasp_model.pth')
            self.config = rospy.get_param('~config', pkg + '/Completion/cfgs/YCB_models/SGrasp.yaml')
            self.use_gpu = True
            self.resume = False
            self.local_rank = 0
            self.experiment_path = pkg + '/experiments'
            pathlib.Path(self.experiment_path).mkdir(parents=True, exist_ok=True)

    def __init__(self) -> None:
        args = self.Arg()
        torch.backends.cudnn.benchmark = True
        config = get_config(args)

        self.__base_model = builder.model_builder(config.model)
        # load checkpoints
        builder.load_model(self.__base_model, args.ckpts)
        self.__base_model.to(args.local_rank)

        # Criterion
        self.__base_model.eval()

        self.__pub = rospy.Publisher('complete_cloud', PointCloud2, queue_size=1)
        rospy.Service('complete_cloud', CompleteCloud, self.__service_callback)
        # rospy.Subscriber('partial_cloud', PointCloud2, self.__callback)

    def __service_callback(self, req: CompleteCloudRequest) -> CompleteCloudResponse:
        with torch.no_grad():
            partial_ = np.asarray(
                [[p[0], p[1], p[2]]
                 for p in pc2.read_points(req.partial, skip_nans=True, field_names=('x', 'y', 'z'))])
            partial = runner.farthest_point_sample(partial_, 2048)
            partial = partial[:, :3]

            centroid = np.mean(partial, axis=0)
            pc = partial - centroid
            m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
            pc = pc / m
            partial = pc.reshape(1, 2048, 3)

            partial = torch.tensor(partial)
            partial = partial.float().cuda()

            ret = self.__base_model(partial)
            dense_points = ret[1]

            pcd = dense_points.cpu().squeeze()

            pcd_x = pcd * (m + (m / 6))
            pcd_x = pcd_x + centroid

            complete_msg = self.__to_point_cloud(pcd_x.cpu().squeeze().numpy(), req.partial.header)

            return CompleteCloudResponse(complete_msg)

    def __callback(self, msg: PointCloud2) -> None:
        with torch.no_grad():
            partial_ = np.asarray([[p[0], p[1], p[2]]
                                   for p in pc2.read_points(msg, skip_nans=True, field_names=('x', 'y', 'z'))])
            partial = runner.farthest_point_sample(partial_, 2048)
            partial = partial[:, :3]

            centroid = np.mean(partial, axis=0)
            pc = partial - centroid
            m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
            pc = pc / m
            partial = pc.reshape(1, 2048, 3)

            partial = torch.tensor(partial)
            partial = partial.float().cuda()

            ret = self.__base_model(partial)
            dense_points = ret[1]

            pcd = dense_points.cpu().squeeze()

            pcd_x = pcd * (m + (m / 6))
            pcd_x = pcd_x + centroid

            complete_msg = self.__to_point_cloud(pcd_x.cpu().squeeze().numpy(), msg.header)
            self.__pub.publish(complete_msg)

    def __to_point_cloud(self, points: np.ndarray, header: Header) -> PointCloud2:
        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize

        data = points.astype(dtype).tobytes()

        fields = [PointField(name=n, offset=i * itemsize, datatype=ros_dtype, count=1) for i, n in enumerate('xyz')]

        return PointCloud2(header=header,
                           height=1,
                           width=points.shape[0],
                           is_dense=False,
                           is_bigendian=False,
                           fields=fields,
                           point_step=(itemsize * 3),
                           row_step=(itemsize * 3 * points.shape[0]),
                           data=data)


if __name__ == '__main__':
    rospy.init_node('completion')
    _ = Completion()
    rospy.spin()
