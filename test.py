from utils import *
import glob
import os
import tensorflow as tf
from argparse import ArgumentParser

parser=ArgumentParser()
parser.add_argument('--gpu', type=str, dest='gpu', default='0', help='The index of GPU which is going to be used')
parser.add_argument('--ref', type=int, dest='ref', default=1, choices=[0,1], help='1: if there exist reference images, 0: No reference images.')
parser.add_argument('--datapath', type=str, dest='datapath', help='Input image data path')
parser.add_argument('--labelpath', type=str, dest='labelpath', default=None, help='Ground truth image data path if available')
parser.add_argument('--modelpath', type=str, dest='modelpath', default='Model', help='Model path')
parser.add_argument('--model', type=str, dest='model',choices=['NatSR', 'FRSR', 'FRSR_x2', 'FRSR_x3'], default='NatSR', help='Model type: NatSR or FRSR ?')
parser.add_argument('--savepath', type=str, dest='savepath', default='result', help='savepath')
parser.add_argument('--save', dest='save', default=False, action='store_true', help='To save output images')
options=parser.parse_args()

conf = tf.ConfigProto()
conf.gpu_options.per_process_gpu_memory_fraction = 0.9

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']=options.gpu

scale = 4
ext='png'

datapath=options.datapath
img_list=np.sort(np.asarray(glob.glob('%s/*.%s' % (datapath, ext))))

if options.ref == 1:
    labelpath=options.labelpath
    label_list = np.sort(np.asarray(glob.glob('%s/*.%s' % (labelpath, ext))))
    P=[]

modelpath=options.modelpath
model=options.model
savepath=options.savepath
save=options.save

fileNum=len(img_list)
for i in img_list:
    print(os.path.basename(i))

print('=========== Processing Model {} ==========='.format(model))

saver=tf.train.import_meta_graph('%s/%s.meta' % (modelpath, model))
InNOut=tf.get_collection('InNOut')

input_img= InNOut[0]
output=InNOut[1]

with tf.Session(config=conf) as sess:
    ckpt_model = os.path.join(modelpath, model)
    print(ckpt_model)
    saver.restore(sess, ckpt_model)

    for n in range(fileNum):
        image = imread(img_list[n])

        img=image[None, :,:, :]
        out=sess.run(output, feed_dict={input_img: img})

        if options.ref == 1:
            label = imread(label_list[n])
            label = modcrop(label, scale)

            '''Label Y-Channel Only'''
            label_y = np.round(label * 255)
            label_y = label_y.astype(np.uint8)
            label_y = rgb2y(label_y)

            '''Output Y-Channel Only'''
            out_y = np.round(np.clip(out[0] * 255, 0., 255.))
            out_y = out_y.astype(np.uint8)
            out_y = rgb2y(out_y)

            P.append(psnr(label_y[scale:-scale,scale:-scale], out_y[scale:-scale,scale:-scale]))


        if save is True:
            if not os.path.exists('%s/%s' % (savepath, model)):
                os.makedirs('%s/%s' % (savepath, model))

            saveImg=np.uint8(np.round(np.clip(out[0]*255,0.,255.)))
            imageio.imsave(os.path.join(savepath, model, os.path.basename(img_list[n][:-4]+'.png')), saveImg)

if options.ref is True:
    print('Y-PSNR: %.2f' % np.mean(P))

print('Done')
