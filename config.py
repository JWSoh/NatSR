from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--trial', type=int, dest='trial', default=0)
parser.add_argument('--gpu', type=str, dest='gpu', default='0')
parser.add_argument('--step', type=int, dest='step', default=0)

OPTIONS=parser.parse_args()

HEIGHT = 192
WIDTH = 192
CHANNEL = 3
BATCH_SIZE = 32
EPOCH = 20000
LEARNING_RATE = 2e-4
TF_RECORD_PATH = 'train_NatSR_X4.tfrecord'
CHECK_POINT_DIR = 'SR'
NUM_OF_DATA = 352432

SCALE = 4