## Distributed training with ray
The `example` contains both tensorflow's and pytorch's examples.

The `timeHistory.py` file is a timer for TensorFlow Distributed training which is not contained in ray.
If you want to set the timer, you can see [here](#Add-Timer)

#### Run `tensorflow` example with command

`python tensorflow_train_example.py -n 2 --hadoop_conf $HADOOP_CONF_DIR/`

`-n` is the number of node you you want to use.

``--hadoop_conf`` is the yarn's config file path. If you want to run it on local, just delete this arg.

Also, you can add `--batch_size` to set batch size, the default value is 128.
 
#### Run `pytorch` example with command
`python train_example.py -n 2 --hadoop_conf $HADOOP_CONF_DIR/`

`-n` is the number of node you you want to use.

``--hadoop_conf`` is the yarn's config file path. If you want to run it on local, just delete this arg.

But in this two files, we didn't ues `init_spark_on_local` and if you need, you can the change code manually.

#### Add Timer 
You can add the `timeHistory.py` file into the path `PATH TO RAY/python/ray/experimental/sgd/tf` and change the file `tf_runner.py`

```python
from ray.experimental.sgd.tf.timeHistory import TimeHistory
...
...
...
    def step(self):
        ...
        time_callback = TimeHistory()
        history = self.model.fit(self.train_dataset, **fit_default_config, callbacks=[time_callback])
        if history is None:
            stats = {}
        else:
            logger.info(time_callback.batch_time)
            stats = {"train_" + k: v[-1] for k, v in history.history.items()}
            stats["batch_time"] = sum(time_callback.batch_time) / len(time_callback.batch_time)
        ...
...
...
```