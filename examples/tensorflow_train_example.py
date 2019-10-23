from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.applications import resnet50

import ray
from ray import tune
from ray.experimental.sgd.tf.tf_trainer import TFTrainer, TFTrainable
from zoo import init_spark_on_yarn, init_spark_on_local
from zoo.ray.util.raycontext import RayContext


NUM_TRAIN_SAMPLES = 3000
NUM_TEST_SAMPLES = 1000


def create_config(batch_size):
    return {
        "batch_size": batch_size,
        "fit_config": {
            "steps_per_epoch": NUM_TRAIN_SAMPLES // batch_size
        },
        "evaluate_config": {
            "steps": NUM_TEST_SAMPLES // batch_size,
        }
    }


def linear_dataset(a=2, size=1000):
   # x = np.random.rand(size)
   # y = x / 2
   #
   # x = x.reshape((-1, 1))
   # y = y.reshape((-1, 1))
    x = np.random.uniform(low=0, high=1, size=[size, 224, 224, 3]).astype(np.float32)
    y = np.random.randint(low=0, high=1000, size=[size]).astype(np.long)

    return x, y


def simple_dataset(config):
    batch_size = config["batch_size"]
    x_train, y_train = linear_dataset(size=NUM_TRAIN_SAMPLES)
    x_test, y_test = linear_dataset(size=NUM_TEST_SAMPLES)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_dataset = train_dataset.shuffle(NUM_TRAIN_SAMPLES).repeat().batch(
        batch_size)
    test_dataset = test_dataset.repeat().batch(batch_size)

    return train_dataset, test_dataset


# def simple_model(config):
#     model = Sequential([Dense(10, input_shape=(1, )), Dense(1)])
#
#     model.compile(
#         optimizer="sgd",
#         loss="mean_squared_error",
#         metrics=["mean_squared_error"])
#
#     return model

def simple_model(config):
    model = resnet50.ResNet50(weights=None)
    # model = Sequential([Dense(10, input_shape=(1, )), Dense(1)])

    model.compile(
       optimizer="Adam",
       loss="categorical_crossentropy",
       metrics=["categorical_crossentropy"])

    return model



def train_example(num_replicas=1, batch_size=128, use_gpu=False):
    trainer = TFTrainer(
        model_creator=simple_model,
        data_creator=simple_dataset,
        num_replicas=num_replicas,
        use_gpu=use_gpu,
        verbose=True,
        config=create_config(batch_size*num_replicas))

    train_stats1 = trainer.train()
    train_stats1.update(trainer.validate())
    print(train_stats1)
    print("Throughput: " + str(batch_size*num_replicas/train_stats1["batch_time"]))

    val_stats = trainer.validate()
    print(val_stats)
    print("success!")


def tune_example(num_replicas=1, use_gpu=False):
    config = {
        "model_creator": tune.function(simple_model),
        "data_creator": tune.function(simple_dataset),
        "num_replicas": num_replicas,
        "use_gpu": use_gpu,
        "trainer_config": create_config(batch_size=128)
    }

    analysis = tune.run(
        TFTrainable,
        num_samples=2,
        config=config,
        stop={"training_iteration": 2},
        verbose=1)

    return analysis.get_best_config(metric="validation_loss", mode="min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Tune training")
    parser.add_argument("--hadoop_conf",
                        type=str,
                        help="turn on yarn mode by passing the hadoop path"
                        "configuration folder. Otherwise, turn on local mode.")

    args, _ = parser.parse_known_args()

    if args.hadoop_conf:
        sc = init_spark_on_yarn(
            hadoop_conf=args.hadoop_conf,
            conda_name="rayexample",
            num_executor=args.num_replicas,
            executor_cores=88,
            executor_memory="10g",
            driver_memory="3g",
            driver_cores=4,
            extra_executor_memory_for_ray="2g")
        ray_ctx = RayContext(
            sc=sc,
            object_store_memory="5g")
        ray_ctx.init()
    else:
        ray.init(redis_address=args.redis_address)
        # sc = init_spark_on_local(cores=44)
        # ray_ctx = RayContext(sc=sc, object_store_memory="5g")

    #ray.init(redis_address=args.redis_address)

    if args.tune:
        tune_example(num_replicas=args.num_replicas, use_gpu=args.use_gpu)
    else:
        train_example(num_replicas=args.num_replicas, batch_size=args.batch_size, use_gpu=args.use_gpu)
