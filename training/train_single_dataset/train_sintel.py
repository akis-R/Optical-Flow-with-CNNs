import os
import sys
import yaml
import argparse
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime

from tf_raft.model import RAFT, SmallRAFT
from tf_raft.losses import sequence_loss, end_point_error
from tf_raft.datasets import MpiSintel, ShapeSetter, CropOrPadder
from tf_raft.training import VisFlowCallback, first_cycle_scaler
import faulthandler



def train(config, logdir):
    try:
        data_config = config['data']
        root = data_config['root']

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
    ds_train = MpiSintel(aug_params,
                         split='training',
                         root=root,
                         dstype='clean')
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
    val_size = int(0.1*train_size)
    ds_val = MpiSintel(split='training',
                       root=root,
                       dstype='clean')
    ds_val.shuffle()
    ds_val.image_list = ds_val.image_list[:val_size]
    ds_val.flow_list = ds_val.flow_list[:val_size]
    print(f'Found {val_size} samples for validation')
    
    ds_val = tf.data.Dataset.from_generator(
        ds_val,
        output_types=(tf.uint8, tf.uint8, tf.float32, tf.bool),
    )
    ds_val = ds_val.batch(1)\
                   .map(ShapeSetter(batch_size=1, image_size=(436, 1024)))\
                   .map(CropOrPadder(target_size=(448, 1024)))\
                   .prefetch(buffer_size=1)

    # for visualization
    ds_vis = MpiSintel(split='training',
                       root=root,
                       dstype='clean')
    ds_vis.shuffle()

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

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=logdir+'/history'),
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

    raft.fit(
        ds_train,
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=train_size//batch_size,
        validation_data=ds_val,
        validation_steps=val_size
    )


if __name__ == '__main__':

    with open('training/configs/train_sintel.yml', 'r') as f:
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
