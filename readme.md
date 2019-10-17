## Distributed training with ray

It is a part of Ray's code. The full ray can checkout `tf` branch.

After you install ray with pip, you can replace or change the files under dir `..your envs path/site-packages/ray/experimental/sgd/tf(or pytorch)` with the files under `sgd`.


And the `example` contains both tensorflow's and pytorch's examples.

#### Run `tensorflow` example with command

`python tensorflow_train_example.py -n 2 --hadoop_conf $HADOOP_CONF_DIR/`

`-n` is the number of node you you want to use.

``--hadoop_conf`` is the yarn's config file path. If you want to run it on local, just delete this arg.

Also, you can add `--batch_size` to set batch size, the default value is 128.
 
#### Run `pytorch` example with command
`python train_example.py -n 2 --hadoop_conf $HADOOP_CONF_DIR/`

`-n` is the number of node you you want to use.

``--hadoop_conf`` is the yarn's config file path. If you want to run it on local, just delete this arg.

But in this two files, we didn't ues `init_spark_on_local` and if you need, you should change code manually.
