"""
This file holds code for a Training guide for PytorchSGD in the documentation.

It ignores yapf because yapf doesn't allow comments right after code blocks,
but we put comments right after code blocks to prevent large white spaces
in the documentation.
"""

# yapf: disable
# __torch_train_example__
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import torch
import torch.nn as nn
import time
import torchvision

from ray.experimental.sgd.pytorch.pytorch_trainer import PyTorchTrainer
from zoo import init_spark_on_yarn, init_spark_on_local
from zoo.ray.util.raycontext import RayContext


class LinearDataset(torch.utils.data.Dataset):
    """y = a * x + b"""

    def __init__(self, size=3000):
        x = torch.from_numpy(np.random.uniform(low=0, high=1, size=[size, 3, 224, 224]).astype(np.float32))
        y = torch.from_numpy(np.random.randint(low=0, high=1000, size=size).astype(np.long))
        print("Y shape: "+str(y.shape))
        print("X shape: "+str(x.shape))
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)



def model_creator(config):
    return torchvision.models.resnet50(pretrained=False)


def optimizer_creator(model, config):
    """Returns criterion, optimizer"""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    return criterion, optimizer


def data_creator(config):
    """Returns training set, validation set"""
    return LinearDataset(), LinearDataset(size=400)


def train_example(num_replicas=1, use_gpu=False):
    batch_size = 512
    import socket
    print(socket.gethostname())
    trainer1 = PyTorchTrainer(
        model_creator,
        data_creator,
        optimizer_creator,
        num_replicas=num_replicas,
        use_gpu=use_gpu,
        batch_size=batch_size*num_replicas,
        backend="gloo")
    sum = 0
    epoch = 10
    for i in range(epoch):
        stas = trainer1.train()
        print("-----Epoch:"+str(i)+"-------")
        print("Stas return : "+ str(stas))
        print("Throughtput :"+str(batch_size*num_replicas/stas["batch_time"]))
        sum += (batch_size*num_replicas/stas["batch_time"])
        print("Data time :"+str(stas["data_time"]))
        print("Training mean :"+str(stas["training_time_mean"]))
        print("Training total :"+str(stas["training_time_total"]))
    print("Mean tp: "+str(sum/epoch))
    trainer1.shutdown()
    print("success!")


if __name__ == "__main__":

    #import ssl

    #ssl._create_default_https_context = ssl._create_unverified_context
    parser = argparse.ArgumentParser()
    parser.add_argument("--hadoop_conf",
                        type=str,
                        help="turn on yarn mode by passing the hadoop path"
                        "configuration folder. Otherwise, turn on local mode.")
    parser.add_argument(
        "--redis-address",
        required=False,
        type=str,
        help="the address to use for Redis")
    parser.add_argument(
        "--num-replicas",
        "-n",
        type=int,
        default=1,
        help="Sets number of replicas for training.")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Enables GPU training")
    parser.add_argument(
        "--tune", action="store_true", default=False, help="Tune training")

    args, _ = parser.parse_known_args()
    import ray

    #ray.init(redis_address=args.redis_address)
    if args.hadoop_conf:
        slave_num = args.num_replicas
        print("Slave num : " + str(slave_num))
        sc = init_spark_on_yarn(
            hadoop_conf=args.hadoop_conf,
            conda_name="rayexample",
            num_executor=slave_num,
            executor_cores=88,#88
            executor_memory="10g",
            driver_memory="5g",
            driver_cores=4,
            extra_executor_memory_for_ray="10g")
        print("Init spark success!")
        ray_ctx = RayContext(sc=sc,object_store_memory="10g")
        print("RayContext success!")
        ray_ctx.init()
    else:
        # sc = init_spark_on_local(cores=22)
        # ray_ctx = RayContext(sc=sc)
        ray.init()

    print("ray init")
    t_s = time.time()
    train_example(num_replicas=args.num_replicas, use_gpu=args.use_gpu)
    t_e = time.time()
    print("Total time: {}.".format(t_e - t_s))
