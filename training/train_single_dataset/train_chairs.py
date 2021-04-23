import os
import sys
import yaml
import argparse
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime

from tf_raft.model import RAFT, SmallRAFT
from tf_raft.losses import sequence_loss, end_point_error
from tf_raft.datasets import FlyingChairs, ShapeSetter, CropOrPadder
from tf_raft.training import VisFlowCallback, first_cycle_scaler


def train(config, logdir):
    try:
        data_config = config['data']
        root = data_config['root']
        split_txt = data_config['split_txt']

        aug_params = config['augment']
        crop_size = aug_params['crop_size']

        model_config = config['model']
        iters = model_config['iters']
        iters_pred = model_config['iters_pred']

        train_config = config['train']
        epochs = train_config['epochs']
        batch_size = train_config['batch_size']
        learning_rate = train_config['learning_rate']
        weight_decay = train_config['weight_decay']
        clip_norm = train_config['clip_norm']

        vis_config = config['visualize']
        num_visualize = vis_config['num_visualize']
        choose_random = vis_config['choose_random']
    except ValueError:
        print('invalid arguments are given')

    # training set
    ds_train = FlyingChairs(aug_params,
                            split='training',
                            split_txt=split_txt,
                            root=root)
    ds_train.shuffle()
    train_size = len(ds_train)
    print(f'Found {train_size} samples for training')

    ds_train = tf.data.Dataset.from_generator(
        ds_train,
        output_types=(tf.uint8, tf.uint8, tf.float32, tf.bool),
    )
    ds_train = ds_train.repeat(epochs)\
                       .batch(batch_size)\
                       .map(ShapeSetter(batch_size, crop_size))\
                       .prefetch(buffer_size=1)

    # validation set
    ds_val = FlyingChairs(split='validation',
                          split_txt=split_txt,
                          root=root)
    val_size = len(ds_val)
    print(f'Found {val_size} samples for validation')
    
    ds_val = tf.data.Dataset.from_generator(
        ds_val,
        output_types=(tf.uint8, tf.uint8, tf.float32, tf.bool),
    )
    ds_val = ds_val.batch(1)\
                   .map(ShapeSetter(batch_size=1, image_size=(384, 512)))\
                   .prefetch(buffer_size=1)

    # for visualization
    ds_vis = FlyingChairs(split='validation',
                          split_txt=split_txt,
                          root=root)    

    scheduler = tfa.optimizers.CyclicalLearningRate(
        initial_learning_rate=learning_rate,
        maximal_learning_rate=2*learning_rate,
        step_size=25000,
        scale_fn=first_cycle_scaler,
        scale_mode='cycle',
    )

    optimizer = tfa.optimizers.AdamW(
        weight_decay=weight_decay,
        learning_rate=scheduler,
    )

    raft = RAFT(drop_rate=0, iters=iters, iters_pred=iters_pred)
    raft.compile(
        optimizer=optimizer,
        clip_norm=clip_norm,
        loss=sequence_loss,
        epe=end_point_error
    )

    # print('Restoring pretrained weights ...', end=' ')
    # raft.load_weights(resume)
    # print('done')
    
    # total_parameters = 0
    # for variable in graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    # # shape is an array of tf.Dimension
    #     shape = variable.get_shape()
    #     print("Size of the matrix: {}".format(shape))
    #     print("How many dimensions it has: {}".format(len(shape)))
    #     variable_parameters = 1
    #     for dim in shape:
    #         print("Dimension: {}".format(dim))
    #         variable_parameters *= dim.value
    #     print("Total number of elements in a matrix: {}".format(variable_parameters))
    #     print("---------------------------------------------")
    #     total_parameters += variable_parameters
    # print("Total number of parameters: {}". format(total_parameters))

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logdir+'/history',
            histogram_freq=1,
            embeddings_freq=0, 
            update_freq="batch",
            write_graph=True
        ),
        VisFlowCallback(
            ds_vis,
            num_visualize=num_visualize,
            choose_random=choose_random,
            logdir=logdir+'/predicted_flows'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=logdir+'/checkpoints/model',
            save_weights_only=True,
            monitor='val_epe',
            mode='min',
            save_best_only=True
        )
    ]

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    raft.fit(
        ds_train,
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=train_size//batch_size,
        validation_data=ds_val,
        validation_steps=val_size
    )

    raft.summary()



if __name__ == '__main__':

    with open('training/configs/train_chairs.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    logd = config['logdir']
    logdir = os.path.join(logd, datetime.now().strftime("%Y-%m-%dT%H-%M"))
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    savepath = logdir + "/config.yml"
    with open(savepath, 'w') as f:
        f.write(yaml.dump(config, default_flow_style=False))

    print('\n ------------------------ Config --------------------------- \n')
    print(yaml.dump(config))
    print('\n ----------------------------------------------------------- \n')
    train(config, logdir)