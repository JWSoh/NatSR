import scipy.misc
import os
import glob
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from .augmentation import augmentation

scale=4

parser=ArgumentParser()
parser.add_argument('--labelpath', dest='labelpath', help='Path to HR images (./DIV2K_train_HR)')
parser.add_argument('--datapath', dest='datapath', help='Path to bicubic interpolated LR images (./DIV2K_train_ILR_bicubic)')
parser.add_argument('--tfrecord', dest='tfrecord', help='Save path for tfrecord file', default='train_NMD_X%d.tfrecord' % scale)
options=parser.parse_args()

labelpath=options.labelpath
datapath=options.datapath
tfrecord_file = options.tfrecord

def imread(path):
    img = scipy.misc.imread(path)
    return img

def gradients(x):
    return np.mean(((x[:-1, :-1, :] - x[1:, :-1, :]) ** 2 + (x[:-1, :-1, :] - x[:-1, 1:, :]) ** 2))

def modcrop(imgs, modulo):
    sz=imgs.shape
    sz=np.asarray(sz)
    if len(sz)==2:
        sz = sz - sz% modulo
        out = imgs[0:sz[0], 0:sz[1]]
    elif len(sz)==3:
        szt = sz[0:2]
        szt = szt - szt % modulo
        out = imgs[0:szt[0], 0:szt[1],:]

    return out

def data_ready_NMD(data_path,label_path,tfrecord_file,patch_h,patch_w,stride):
    label_list=np.sort(np.asarray(glob.glob(os.path.join(label_path, '/*.png'))))
    img_list = np.sort(np.asarray(glob.glob(os.path.join(data_path, 'X' +  str(scale) + '/*.png'))))

    offset=0

    fileNum=len(label_list)

    patches=[]
    labels=[]

    for n in range(fileNum):
        img=imread(img_list[n])
        label=imread(label_list[n])

        assert os.path.basename(img_list[n])[:-6] == os.path.basename(label_list[n])[:-4]

        x,y,ch=label.shape
        for m in range(4):
            for i in range(0+offset,x-patch_h+1,stride):
                for j in range(0+offset,y-patch_w+1,stride):
                    patch_d = img[i:i + patch_h, j:j+ patch_w]
                    patch_l = label[i:i + patch_h, j:j + patch_w]

                    if gradients(patch_l.astype(np.float64)/255.) >= 0.005 and np.var(patch_l.astype(np.float64)/255.) >= 0.03:
                        patches.append(augmentation(patch_d, m).tobytes())
                        labels.append(augmentation(patch_l, m).tobytes())


    np.random.seed(36)
    np.random.shuffle(patches)
    np.random.seed(36)
    np.random.shuffle(labels)
    print(len(patches))
    print('Shape: [%d, %d, %d]' % (patch_h, patch_w, ch) )

    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    for i in range(len(patches)):
         write_to_tfrecord(writer, labels[i], patches[i])

    writer.close()


def write_to_tfrecord(writer, label, image):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
        }))
    writer.write(example.SerializeToString())
    return

if __name__=='__main__':
    data_ready_NMD(datapath, labelpath, tfrecord_file,48*scale,48*scale,120)
    print('Done')