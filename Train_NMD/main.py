from config import *
import os
import tensorflow as tf
import train

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=OPTIONS.gpu

conf=tf.ConfigProto()
conf.gpu_options.per_process_gpu_memory_fraction=0.95

def main():
    Train=train.Train(trial=OPTIONS.trial, step=OPTIONS.step, size=[HEIGHT, WIDTH, CHANNEL], batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE, max_epoch=EPOCH, tfrecord_path=TF_RECORD_PATH,checkpoint_dir=CHECK_POINT_DIR,
                          num_of_data=NUM_OF_DATA,conf=conf)
    Train()
    # python main.py --gpu 0 --trial 1 --step 0

if __name__=='__main__':
    main()