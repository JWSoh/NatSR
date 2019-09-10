from utils import *
import model

class Train(object):
    def __init__(self, trial, step, size, batch_size, learning_rate, max_epoch, tfrecord_path, checkpoint_dir, num_of_data, conf):
        print('[*] Initialize Training')
        self.trial = trial
        self.step=step
        self.HEIGHT=size[0]
        self.WIDTH=size[1]
        self.CHANNEL=size[2]
        self.BATCH_SIZE=batch_size
        self.learning_rate=learning_rate
        self.EPOCH=max_epoch
        self.tfrecord_path=tfrecord_path
        self.checkpoint_dir=checkpoint_dir
        self.num_of_data=num_of_data
        self.conf=conf

        self.image = tf.placeholder(tf.float32, shape=[None, self.HEIGHT,self.WIDTH, self.CHANNEL])
        self.label = tf.placeholder(tf.float32, shape=[None, None])

        self.MODEL=model.NMD(self.image)

    def calc_loss(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label,logits=self.MODEL.logit))

    def __call__(self):
        print('[*] Setting Train Configuration')

        self.calc_loss()

        '''Learning rate and Global step'''
        self.learning_rate=tf.constant(self.learning_rate)
        self.global_step=tf.Variable(self.step, name='global_step', trainable=False)

        '''Optimizer'''
        self.opt= tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)

        '''Variables'''
        self.sigma = tf.Variable(0.1, trainable=False, name='noise_sigma')
        self.alpha = tf.Variable(0.5, trainable=False, name='alpha')

        self.accuracy=accuracy_calc(predictions=self.MODEL.output, labels=self.label)

        '''Summary'''
        self.train_summary = tf.summary.merge([tf.summary.scalar("total_loss", self.loss),
                                               tf.summary.scalar("train_acc", self.accuracy),
                                               tf.summary.image("input", tf.clip_by_value(self.image, 0., 1.), max_outputs=4),
                                               tf.summary.scalar("sigma", self.sigma),
                                               tf.summary.scalar("alpha", self.alpha)])
        self.val_summary1 = tf.summary.merge([tf.summary.scalar('val_sigma_acc', self.accuracy)])
        self.val_summary2 = tf.summary.merge([tf.summary.scalar('val_alpha_acc', self.accuracy)])

        '''Update Assignments'''
        update_sigma = tf.assign(self.sigma, self.sigma * 0.8)
        update_alpha = tf.assign(self.alpha, tf.clip_by_value(self.alpha + 0.1, 0., 0.9))

        '''Training'''
        self.init = tf.global_variables_initializer()
        self.saver=tf.train.Saver(max_to_keep=100000)

        '''Load TFRECORD dataset'''
        input_train, bic_train = self.load_tfrecord(self.tfrecord_path, self.BATCH_SIZE)
        val_train, val_bic = self.load_tfrecord('Val_2521.tfrecord', 200)

        with tf.Session(config=self.conf) as sess:
            sess.run(self.init)

            could_load, model_step=load(self.saver,sess, self.checkpoint_dir, folder='Model%d' % self.trial)
            if could_load:
                print('Iteration:', self.step)
                print('==================== Load Succeeded ====================')
                assert self.step == model_step, 'The latest step and the input step do not match.'
            else:
                print('==================== No model to load ====================')

            train_writer = tf.summary.FileWriter('./logs%d/train' % self.trial, sess.graph)
            val_writer = tf.summary.FileWriter('./logs%d/val' % self.trial, sess.graph)

            print('[*] Training Starts')
            step=self.step
            num_of_batch = self.num_of_data // self.BATCH_SIZE
            s_epoch = (step*self.BATCH_SIZE) // self.num_of_data

            epoch=s_epoch

            val_stack1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            val_stack2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            while True:
                try:
                    '''Load Data'''
                    input_train_, bic_train_ = sess.run([input_train, bic_train])

                    '''Noise injection + image interpolation'''
                    input_noisy = inject_dct(input_train_[0:self.BATCH_SIZE // 4], sigma=sess.run(self.sigma))
                    input_inter = interpolate_img(input_train_[self.BATCH_SIZE // 4:self.BATCH_SIZE // 2],
                                                  bic_train_[self.BATCH_SIZE // 4:self.BATCH_SIZE // 2], sess.run(self.alpha))
                    input_clean = input_train_[self.BATCH_SIZE // 2:]

                    input_TRAIN = np.concatenate([input_noisy, input_inter, input_clean], axis=0)
                    input_LABEL = np.concatenate([np.zeros([self.BATCH_SIZE // 2, 1]), np.ones([self.BATCH_SIZE // 2, 1])],
                                                 axis=0)

                    sess.run(self.opt, feed_dict={self.image: input_TRAIN, self.label: input_LABEL})

                    step += 1

                    if step % 100 == 0:
                        val_train_, val_bic_ = sess.run([val_train, val_bic])

                        val_noisy = inject_dct(val_train_[0:100], sigma=sess.run(self.sigma))
                        val_inter = interpolate_img(val_train_[0:100], val_bic_[0:100], sess.run(self.alpha))
                        val_clean = val_train_[100:]

                        val_TRAIN_1 = np.concatenate([val_noisy, val_clean], axis=0)
                        val_TRAIN_2 = np.concatenate([val_inter, val_clean], axis=0)

                        val_LABEL = np.concatenate([np.zeros([100, 1]), np.ones([100, 1])], axis=0)

                        loss_, train_summary_, accuracy_ = sess.run([self.loss, self.train_summary, self.accuracy],
                                                                    feed_dict={self.image: input_TRAIN, self.label: input_LABEL})

                        val_summary1_, val_accuracy1_ = sess.run([self.val_summary1, self.accuracy],
                                                               feed_dict={self.image: val_TRAIN_1, self.label: val_LABEL})
                        val_summary2_, val_accuracy2_ = sess.run([self.val_summary2, self.accuracy],
                                                                 feed_dict={self.image: val_TRAIN_2, self.label: val_LABEL})

                        val_stack1.append(val_accuracy1_)
                        val_stack2.append(val_accuracy2_)

                        val_stack1 = val_stack1[1:]
                        val_stack2 = val_stack2[1:]
                        val_avg1 = np.mean(val_stack1)
                        val_avg2 = np.mean(val_stack2)

                        print('Iteration:', step, 'Loss:', loss_, 'Train_Acc:', accuracy_, 'Val_Acc:', val_accuracy1_,
                              val_accuracy2_, 'Sigma:', sess.run(self.sigma), 'Alpha:', sess.run(self.alpha))

                        train_writer.add_summary(train_summary_, step)
                        train_writer.flush()
                        val_writer.add_summary(val_summary1_, step)
                        val_writer.add_summary(val_summary2_, step)
                        val_writer.flush()

                        if val_avg1 >= 95.0:
                            print('[*******] Iteration:', step, 'Scale Sigma')
                            sess.run(update_sigma)

                        if val_avg2 == 95.0:
                            print('[*******] Iteration:', step, 'Scale alpha')
                            sess.run(update_alpha)


                    if step % 10000 == 0:
                        save(self.saver, sess, self.checkpoint_dir, self.trial, step)

                    if step % num_of_batch == 0:
                        print('[*] Epoch:', epoch, 'Done')
                        epoch += 1

                        if epoch == self.EPOCH:
                            break

                        print('[*] Epoch:', epoch, 'Starts', 'Total iteration', step)

                except KeyboardInterrupt:
                    print('***********KEY BOARD INTERRUPT *************')
                    print('Epoch:', epoch, 'Iteration:', step)
                    save(self.saver, sess, self.checkpoint_dir, self.trial, step)
                    break

    '''Functions for Loading TFRECORD'''
    def _parse_function(self, example_proto):
        keys_to_features = {'label': tf.FixedLenFeature([], tf.string), 'image': tf.FixedLenFeature([], tf.string)}

        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        label = parsed_features['label']
        img = parsed_features['image']

        label = tf.divide(tf.cast(tf.decode_raw(label, tf.uint8), tf.float32), 255.)
        label = tf.reshape(label, [self.HEIGHT, self.WIDTH, self.CHANNEL])

        img = tf.divide(tf.cast(tf.decode_raw(img, tf.uint8), tf.float32), 255.)
        img = tf.reshape(img, [self.HEIGHT, self.WIDTH, self.CHANNEL])

        return label, img
    def load_tfrecord(self, tfrecord_path, batch_size):
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        dataset = dataset.map(self._parse_function)

        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()

        input_train, bic_train= iterator.get_next()

        return input_train, bic_train