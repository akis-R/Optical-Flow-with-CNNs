import numpy as np
import numpy as np
import cv2
# from tf_raft.losses import photometric_loss
from tf_raft.model import RAFT
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime


def photometric_loss(im1, im2, flow):
    flow = flow[-1]
    im1_w = tfa.image.dense_image_warp(im1, flow=flow)
    im1_flat = tf.reshape(im1_w, [-1])
    im2_flat = tf.reshape(im2, [-1])
    c = tf.stack([im1_flat, im2_flat], axis=1)
    loss = tf.norm(c, ord=1, axis=1, keepdims=None)
    return tf.reduce_sum(loss)/loss.shape[0]

if __name__ == '__main__':

    iters = 4
    iters_pred = 4
    raft = RAFT(iters=iters, iters_pred=iters_pred)
    raft.load_weights('/Users/flo/Google Drive/Augmenta/tf-raft-master/checkpoints/model')
    total_parameters = 0
    cap = cv2.VideoCapture('RGB_mini.mp4')
    # width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out = cv2.VideoWriter(
        filename='visualisation_optical_flow.mp4',
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=cap.get(cv2.CAP_PROP_FPS),
        frameSize=(2001, 2001),
        isColor=True
    )
    ret, frame = cap.read()
    frame = frame[400:1400, 52:2052, :]
    prvs = frame.astype(np.float32)
    prvs.shape = (1, prvs.shape[0], prvs.shape[1], prvs.shape[2])
    hsv = np.zeros_like(frame)
    hsv[...,1] = 255
    flow = 0
    loss_list = list()

    while(True):
        ret, frame = cap.read()
        if np.any(frame)==None:
            np.savetxt("photometric_loss.csv", np.array(loss_list), delimiter=",")
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            break
        frame = frame[400:1400, 52:2052, :]
        # cv2.imshow('original', frame)
        frame.shape = (1, frame.shape[0], frame.shape[1], frame.shape[2])
        frame = frame.astype(np.float32)
        flow = raft([prvs, frame], training=False)        
        loss = photometric_loss(prvs, frame, flow)
        print(str(datetime.now())+': photometric_loss:', loss.numpy())
        loss_list.append(loss)
        prvs = frame
        # frame.shape = (frame.shape[1], frame.shape[2], frame.shape[3])
        flow = flow[-1]
        flow = np.squeeze(flow)
        mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = np.array(cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR), dtype=np.uint8)
        frame = np.array(frame, dtype=np.uint8)
        frame.shape = rgb.shape
        comb = np.concatenate((frame, rgb), axis=0)
        out.write(comb)
        # cv2.imshow('frame',rgb)
