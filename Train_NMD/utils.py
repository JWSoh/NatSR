import numpy as np
import tensorflow as tf
import os
import math
import scipy.misc
import scipy.fftpack

def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    img=img/255.
    return img

def accuracy_calc(predictions, labels):
    equality = tf.equal(tf.cast(tf.greater(predictions, 0.5),tf.float32), labels)
    return 100 * tf.reduce_mean(tf.cast(equality, tf.float32))

def psnr(img1, img2):
    img1=np.float64(img1)
    img2=np.float64(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    if np.max(img1) <= 1.0:
        PIXEL_MAX= 1.0
    else:
        PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def save(saver, sess, checkpoint_dir, trial, step):
    model_name = "model"
    checkpoint_dir = os.path.join(checkpoint_dir, 'Model%d'% trial)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

def load(saver, sess, checkpoint_dir, folder):
    import re
    print('==================== Reading Checkpoints ====================')
    checkpoint = os.path.join(checkpoint_dir, folder)

    ckpt= tf.train.get_checkpoint_state(checkpoint)
    '''ckpt.model_checkpoint_path는 최신 // ckpt.all_model_checkpoint_paths는 학습하는 과정에서 저장한 모든 파일'''

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint, ckpt_name))

        step = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))

        print("=================== Success to load {} ====================".format(ckpt_name))
        return True, step
    else:
        print("=================== Fail to find a Checkpoint ====================")
        return False, 0


def inject_dct(x, sigma):
    n, h, w, c = x.shape
    X_space = np.reshape(x, [n, h // 8, 8, w // 8, 8, c])
    X_dct_x = scipy.fftpack.dct(X_space, axis=2, norm='ortho')
    X_dct = scipy.fftpack.dct(X_dct_x, axis=4, norm='ortho')


    noise_raw= np.random.randn(n, h // 8, 8, w // 8, 8, c) * sigma
    z=np.zeros([n,h//8,8,w//8,8,c])
    z[:, :, 7, :, :, :] = noise_raw[:, :, 7, :, :, :]
    z[:, :, :, :, 7, :] = noise_raw[:, :, :, :, 7, :]

    X_dct_noise = X_dct + z

    Y_space_x = scipy.fftpack.idct(X_dct_noise, axis=2, norm='ortho')
    Y_space = scipy.fftpack.idct(Y_space_x, axis=4, norm='ortho')
    Y = np.reshape(Y_space, x.shape)

    return Y

def interpolate_img(x,y, alpha):
    return alpha*x + (1-alpha)*y

def rgb2y(x):
    if x.dtype==np.uint8:
        x=np.float64(x)
        y=65.481/255.*x[:,:,0]+128.553/255.*x[:,:,1]+24.966/255.*x[:,:,2]+16
        y=np.round(y).astype(np.uint8)
    else:
        y = 65.481 / 255. * x[:, :, 0] + 128.553 / 255. * x[:, :, 1] + 24.966 / 255. * x[:, :, 2] + 16 /255

    return y