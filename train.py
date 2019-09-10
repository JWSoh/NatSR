import model
from utils import *

class Train(object):
    def __init__(self, trial, step, size, batch_size, learning_rate, max_epoch, tfrecord_path, checkpoint_dir, scale,num_of_data, conf):

        print('Initialize Training')
        self.trial=trial
        self.step=step
        self.HEIGHT = size[0]
        self.WIDTH = size[1]
        self.CHANNEL = size[2]
        self.BATCH_SIZE = batch_size
        self.learning_rate=learning_rate
        self.EPOCH = max_epoch
        self.tfrecord_path = tfrecord_path
        self.checkpoint_dir=checkpoint_dir
        self.scale= scale
        self.num_of_data=num_of_data
        self.conf=conf

        self.input = tf.placeholder(dtype=tf.float32,shape=[None,self.HEIGHT//self.scale,self.WIDTH//self.scale,self.CHANNEL])
        self.label = tf.placeholder(dtype=tf.float32,shape=[None,self.HEIGHT,self.WIDTH,self.CHANNEL])

        self.GEN = model.FRACTAL(self.input,self.scale)
        self.NAT = model.NMD(self.GEN.output)
        self.DIS_fake = model.Discriminator(self.GEN.output, reuse=False)
        self.DIS_real = model.Discriminator(self.label, reuse=True)

    def calc_loss(self):
        self.recon_loss=tf.losses.absolute_difference(self.GEN.output , self.label)
        self.nat_loss = -tf.reduce_mean(tf.log(self.NAT.out+1e-10))

        f_logit= self.DIS_fake.logit
        r_logit=self.DIS_real.logit

        '''Relativistic average Standard GAN loss'''
        d_loss_real= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(r_logit), logits=r_logit-tf.reduce_mean(f_logit)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(f_logit), logits=f_logit-tf.reduce_mean(r_logit)))
        g_loss_fake= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(f_logit), logits=f_logit-tf.reduce_mean(r_logit)))
        g_loss_real= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(r_logit), logits=r_logit-tf.reduce_mean(f_logit)))

        self.d_loss= d_loss_fake+d_loss_real
        self.g_loss= g_loss_fake+g_loss_real

        '''Overall Loss'''
        self.loss=self.recon_loss + 1e-3*self.nat_loss + 1e-3 * self.g_loss

    def run(self):
        print('Setting Train Configuration')

        self.calc_loss()

        gen_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='GEN')
        nat_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='CLASSIFIER')
        dis_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='DIS')

        '''Learning rate and the optimizer'''
        self.global_step=tf.Variable(self.step, name='global_step', trainable=False)

        self.lr=tf.train.exponential_decay(self.learning_rate,self.global_step, 100000, 0.1, staircase=True)
        self.lr=tf.maximum(self. lr,1e-4)

        '''Optimizers'''
        self.g_opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=self.global_step, var_list=gen_vars)
        self.d_opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.d_loss, global_step=self.global_step, var_list=dis_vars)

        '''Summary operator for Tensorboard'''
        self.summary_op=tf.summary.merge([tf.summary.scalar('loss', self.loss),
                                          tf.summary.scalar('recon_loss', self.recon_loss),
                                          tf.summary.scalar('naturalness_loss', self.nat_loss),
                                          tf.summary.scalar('natural_score',tf.reduce_mean(self.NAT.out)),
                                          tf.summary.scalar('g_loss', self.g_loss),
                                          tf.summary.scalar('d_loss', self.d_loss),
                                          tf.summary.image('Input', tf.clip_by_value(self.GEN.input, 0., 1.) ,max_outputs=4),
                                          tf.summary.image('Output',tf.clip_by_value(self.GEN.output,0., 1.) ,max_outputs=4)])

        '''Training'''
        for i in gen_vars:
            print(i.name)

        self.loader1=tf.train.Saver(var_list=gen_vars)
        self.loader2=tf.train.Saver(var_list=nat_vars)

        self.saver = tf.train.Saver(max_to_keep=10000, var_list=gen_vars+dis_vars)
        self.init = tf.global_variables_initializer()

        with tf.Session(config=self.conf) as sess:
            sess.run(self.init)

            self.loader1.restore(sess,'Model/FRSR')
            print('Finetune from FRSR')
            self.loader2.restore(sess,'Model/NMD')
            print('Load NMD')

            could_load = load(self.saver, sess, self.checkpoint_dir, folder='Model%d' % self.trial)
            if could_load:
                print('Iteration:', self.step)
                print(' =========== Load Succeeded ============')
            else:
                print(" ========== No model to load ===========")

            train_writer = tf.summary.FileWriter('./logs%d' % self.trial, sess.graph)

            print('Training Starts')
            label_train, input_train = self.load_tfrecord()

            step = self.step

            iter_D=1
            num_of_batch = self.num_of_data // (self.BATCH_SIZE * iter_D)
            s_epoch = (step * self.BATCH_SIZE) // self.num_of_data


            epoch=s_epoch
            while True:
                try:

                    for i in range(iter_D):
                        label_train_, input_train_ = sess.run([label_train, input_train])
                        sess.run(self.d_opt, feed_dict={self.input: input_train_, self.label: label_train_})

                    sess.run(self.g_opt, feed_dict={self.input: input_train_, self.label: label_train_})
                    step = step + 1

                    if step % 1000 == 0:
                        loss_, summary, recon_loss_, nat_loss_, g_loss_, d_loss_ = sess.run([self.loss, self.summary_op, self.recon_loss, self.nat_loss, self.g_loss,self.d_loss],
                            feed_dict={self.input: input_train_, self.label: label_train_})

                        print('Iteration:', step, 'Loss:', loss_, 'Recon_loss:', recon_loss_, 'Nat_loss:', nat_loss_,'G:', g_loss_, 'D:', d_loss_)

                        train_writer.add_summary(summary, step)
                        train_writer.flush()

                    if step % 10000 == 0:
                        save(self.saver, sess, self.checkpoint_dir, self.trial, step)

                    if step % num_of_batch == 0:
                        print('[*] Epoch:', epoch, 'Done')
                        epoch=epoch+1

                        if epoch == self.EPOCH:
                            break

                        print('[*] Epoch:', epoch, 'Starts', 'Total iteration', step)


                except KeyboardInterrupt:
                    print('***********KEY BOARD INTERRUPT *************')
                    print('Epoch:', epoch, 'Iteration:', step)
                    save(self.saver, sess, self.checkpoint_dir, self.trial, step)
                    break


    '''Load TFRECORD'''
    def _parse_function(self, example_proto):
        keys_to_features = {'label': tf.FixedLenFeature([], tf.string),
                            'image': tf.FixedLenFeature([], tf.string)}

        parsed_features = tf.parse_single_example(example_proto, keys_to_features)

        img = parsed_features['image']
        img = tf.divide(tf.cast(tf.decode_raw(img, tf.uint8), tf.float32), 255.)
        img=  tf.reshape(img,[self.HEIGHT//self.scale,self.WIDTH//self.scale,self.CHANNEL])

        label = parsed_features['label']
        label = tf.divide(tf.cast(tf.decode_raw(label, tf.uint8), tf.float32), 255.)
        label = tf.reshape(label, [self.HEIGHT, self.WIDTH, self.CHANNEL])

        return label, img

    def load_tfrecord(self):
        dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        dataset = dataset.map(self._parse_function)

        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.BATCH_SIZE)
        iterator = dataset.make_one_shot_iterator()

        label_train, input_train = iterator.get_next()

        return label_train, input_train