from __future__ import division
import sklearn
import os
import time
from glob import glob
import tensorflow as tf
#from dataset.mnist import load_mnist
import numpy as np
from sklearn.utils import shuffle
from module import *
from utils import *
import utils
import cv2

#TODO : add noise input eta

ATT_ID = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
          'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
          'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
          'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
          'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
          'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
          'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
          'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
          'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
          'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
          'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
          'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
          'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}
ID_ATT = {v: k for k, v in ATT_ID.items()}
att_names = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
                     'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']

class pagan(object):
    
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.ckpt_dir = args.ckpt_dir
        self.gamma = 10
        self.mse = mse_criterion
        self.criterionGAN = sce_criterion
        self.criterionWGAN = wgan_criterion
        self.att_names = args.att_names
        self._build_model(args)
        dir_path = "D:/Experimental/2020/paper_implementation/PAGAN/Original/data/"
        self.saver = tf.train.Saver(max_to_keep=100)
        
        self.img_path, self.labels = self.load_data(dir_path)

    def load_data(self, dir_path):
        img_dir_path = dir_path + "img_align_celeba/img_align_celeba/"
        img_name = np.genfromtxt(dir_path + "train_label.txt", dtype=str, usecols=0)
        img_path = [img_dir_path + i for i in img_name]
        labels = np.genfromtxt(dir_path + "train_label.txt", dtype=int, usecols=range(1,41))
        #labels = np.genfromtxt(label_path, dtype=int, usecols=range(1, 41))
        labels = labels[:, np.array([ATT_ID[att_name] for att_name in self.att_names])]
        return img_path, labels

    def _load_batch(self, img_path, labels, idx):
        load_size = 143
        crop_size = 128
        img_batch = []
        label_batch = []
        for i in range(self.batch_size):
            img = cv2.imread(img_path[i+idx*self.batch_size])
            #img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            img = cv2.resize(img, (load_size,load_size))
            if np.random.random() < 0.5:
                img = cv2.flip(img, 1)
            img = get_random_crop(img, crop_size, crop_size)
            img = img/127.5 - 1
            #img = img/255s
            img_batch.append(img)
            temp_label = [int(j) for j in labels[i+idx*self.batch_size]]
            temp_label = (temp_label+np.ones_like(temp_label))//2
            label_batch.append(temp_label)
        return img_batch, label_batch


    def _build_model(self, args):

        self.real_img = tf.placeholder(tf.float32, [None,128,128,3], name='input')
        self.input_label = tf.placeholder(tf.float32, [None,13], name='label')
        self.is_training = tf.placeholder(tf.bool, [None], name='is_training')
        self.input_label_shf = tf.placeholder(tf.float32, [None,13], name='label_shf')
        

        ##
        g_network = G_network()
        self.d_network = D_network()
        self.input_label_ = self.input_label * 2 - 1
        self.input_label_shf_ = self.input_label_shf * 2 - 1
        #self.label_subtract = tf.subtract(self.input_label_shf_,self.input_label_)
        self.label_subtract = self.input_label_shf_ - self.input_label_
        #self.fake_img, self.mask, self.ms_multi, self.fa_in, self.ek, self.d_m, self.b_atten = g_network(self.real_img, self.label_subtract)
        #self.fake_img, self.mask, ms_multi, self.fa_in = g_network(self.real_img, label_subtract)
        self.fake_img, self.mask, ms_multi = g_network(self.real_img, self.label_subtract)
        d_real, c_real = self.d_network(self.real_img, reuse=False)
        d_fake, c_fake = self.d_network(self.fake_img, reuse=True)


        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        #self.c_vars = [var for var in t_vars if 'attribute_classifier' in var.name]
        print("trainable variables (all) : ")
        print(t_vars)
        print("trainable variables (discriminator) : ")
        print(self.d_vars)
        print("trainable variables (generator) : ")
        print(self.g_vars)
        
        # losses
        
        l_reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        
        

        #self.l_c = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.input_label,logits=c_real)
        self.l_att_d = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.input_label,logits=c_real)
        self.l_adv_d = self.criterionWGAN(d_fake, d_real) + 10*self.gradient_penalty()
        #self.l_reg_d = tf.losses.get_regularization_loss(scope='discriminator')
        self.l_reg_d = [l for l in l_reg if "discriminator" in l.name]

        self.l_att_g = 20*tf.losses.sigmoid_cross_entropy(multi_class_labels=self.input_label_shf,logits=c_fake)
        self.l_adv_g = -tf.reduce_mean(d_fake)
        #self.l_adv_g = tf.losses.sigmoid_cross_entropy(tf.ones_like(d_fake), d_fake)
        self.l_spa = tf.reduce_sum([tf.reduce_mean(m) * w for m, w in zip(self.mask, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])])
        self.l_spa = 0.05*self.l_spa
        self.l_full_ovl, self.l_none_ovl = overlap_loss_fn(ms_multi, self.att_names)
        #self.l_reg_g = tf.losses.get_regularization_loss(scope='generator')
        self.l_reg_g = [l for l in l_reg if "generator" in l.name]

        #self.g_loss = self.l_adv_g + self.l_spa + \
        #    self.l_full_ovl + \
        #    self.l_none_ovl
        self.g_loss = self.l_att_g + \
            self.l_adv_g + self.l_spa + \
            self.l_full_ovl + \
            self.l_none_ovl
        self.g_loss = tf.add(self.g_loss, tf.reduce_sum(self.l_reg_g))

        ##self.d_loss = self.l_c + self.l_adv_d + self.l_att_d
        self.d_loss = self.l_adv_d + self.l_att_d
        self.d_loss = tf.add(self.d_loss, tf.reduce_sum(self.l_reg_d))
        self.loss_summary = tf.summary.scalar("loss", self.d_loss)
        

    def train(self, args):
        
        #self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.lr = args.lr
        
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.lr, global_step, args.epoch_step, 1, staircase=False)

        self.D_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1, beta2=args.beta2) \
            .minimize(self.d_loss, var_list=[self.d_vars], global_step = global_step)
        self.G_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1, beta2=args.beta2) \
            .minimize(self.g_loss, var_list=[self.g_vars], global_step = global_step)
        #self.G_optim_att = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1, beta2=args.beta2) \
        #    .minimize(self.l_att_g, var_list=[self.g_vars], global_step = global_step)
        
        print("initialize")
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 0
        start_time = time.time()


        for epoch in range(args.epoch):
            counter += 1
            batch_idxs = len(self.labels) // self.batch_size

            self.img_path, self.labels = shuffle(self.img_path, self.labels)
            
            for idx in range(0, batch_idxs):

                img_batch, label_batch = self._load_batch(self.img_path, self.labels, idx)
                label_batch_shf = shuffle(label_batch)
                #print(np.subtract(label_batch_shf,label_batch))
                #print(np.max(np.subtract(label_batch_shf,label_batch)))
                _ = self.sess.run([self.D_optim], feed_dict={self.real_img:img_batch, self.input_label:label_batch, self.input_label_shf:label_batch_shf})
                _ = self.sess.run([self.D_optim], feed_dict={self.real_img:img_batch, self.input_label:label_batch, self.input_label_shf:label_batch_shf})
                _ = self.sess.run([self.D_optim], feed_dict={self.real_img:img_batch, self.input_label:label_batch, self.input_label_shf:label_batch_shf})
                _, d_loss,l_adv_d, lr_ = self.sess.run([self.D_optim, self.d_loss,self.l_adv_d, learning_rate], feed_dict={self.real_img:img_batch, self.input_label:label_batch, self.input_label_shf:label_batch_shf})
                
                #_ = self.sess.run([self.G_optim_att], feed_dict={self.real_img:img_batch, self.input_label:label_batch, self.input_label_shf:label_batch_shf})
                fake_img, _, g_loss,l_att_g,l_adv_g,l_spa,l_full_ovl,l_none_ovl = self.sess.run([self.fake_img, self.G_optim, self.g_loss,self.l_att_g,self.l_adv_g,self.l_spa,self.l_full_ovl,self.l_none_ovl], feed_dict={self.real_img:img_batch, self.input_label:label_batch, self.input_label_shf:label_batch_shf})
                #self.writer.add_summary(summary_str, counter)

                #counter += 1
                if idx%1==0:
                    print(("Epoch: [%2d] [%4d/%4d] | D adv loss: %4.4f | G adv loss: %4.4f | time: %4.2f | lr : %4.4f" % (
                        epoch, idx, batch_idxs, d_loss, g_loss, time.time() - start_time, lr_)))
                    print(("Epoch: [%2d] [%4d/%4d] | D adv loss: %4.4f | G adv loss: %4.4f,%4.4f,%4.4f,%4.4f,%4.4f | time: %4.2f" % (
                        epoch, idx, batch_idxs, l_adv_d, l_att_g,l_adv_g,l_spa,l_full_ovl,l_none_ovl, time.time() - start_time)))

                if idx%10 == 0:
                    temp_fake = (fake_img[0]+1)*127.5
                    #cv2.imwrite('./sample/fake_e'+str(epoch)+str(idx)+'.bmp', temp_fake)
                    modi_att = np.random.randint(0,13)
                    #label_batch_shf = label_batch.copy()
                    #label_batch_shf[0][modi_att] = 2
                    #label_batch_shf = shuffle(label_batch)
                    #fake_img, temp_label, mask, fa_in, ek, dm, b_atten = self.sess.run([self.fake_img, self.input_label, self.mask, self.fa_in, self.ek, self.d_m,self.b_atten], feed_dict={self.real_img:img_batch, self.input_label:label_batch, self.input_label_shf:label_batch_shf})
                    fake_img, mask = self.sess.run([self.fake_img, self.mask], feed_dict={self.real_img:img_batch, self.input_label:label_batch, self.input_label_shf:label_batch_shf})
                    temp_fake = (fake_img[0]+1)*127.5
                    ## check
                    #mask_o = (mask[2][0]*255)
                    #mask_o0 = (mask[0][0]*255)
                    #mask_o1 = (mask[1][0]*255)
                    mask_o = (mask[2][0]+1)*127.5
                    mask_o0 = (mask[0][0]+1)*127.5
                    mask_o1 = (mask[1][0]+1)*127.5

                    #fa_in_o = (fa_in[2][0]+1)*127.5
                    #ek_o = (ek[2][0]+1)*127.5
                    #dm_o = np.mean(dm[0], axis=2)*255
                    cv2.imwrite('./sample/fake_e'+str(epoch)+str(idx)+att_names[modi_att]+'.bmp', temp_fake)
                    cv2.imwrite('./sample/mask_'+str(epoch)+str(idx)+att_names[modi_att]+'0.bmp', mask_o0)
                    cv2.imwrite('./sample/mask_'+str(epoch)+str(idx)+att_names[modi_att]+'1.bmp', mask_o1)
                    cv2.imwrite('./sample/mask_'+str(epoch)+str(idx)+att_names[modi_att]+'2.bmp', mask_o)
                    #cv2.imwrite('./sample/fa_in_'+str(epoch)+str(idx)+att_names[modi_att]+'.bmp', fa_in_o)
                    #cv2.imwrite('./sample/ek_'+str(epoch)+str(idx)+att_names[modi_att]+'.bmp', ek_o)
                    #cv2.imwrite('./sample/dm_'+str(epoch)+str(idx)+att_names[modi_att]+'.bmp', dm_o)
                if idx == batch_idxs-1 or idx%int(batch_idxs/4) == 0:
                    self.save(args.checkpoint_dir, counter)


    def save(self, checkpoint_dir, step):
        model_name = "dnn.model"
        model_dir = "%s" % (self.ckpt_dir)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s" % (self.ckpt_dir)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt)
            ckpt_paths = ckpt.all_model_checkpoint_paths    #hcw
            print(ckpt_paths)
            ckpt_name = os.path.basename(ckpt_paths[-1])    #hcw # default [-1]
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


    def test(self, args):

        start_time = time.time()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        counter = 0
        #self.test_image, self.test_label = shuffle(self.test_image, self.test_label)
        for epoch in range(1):
            #batch_idxs = len(self.labels) // self.batch_size
            batch_idxs = 1
            #self.img_path, self.labels = shuffle(self.img_path, self.labels)
            
            for idx in range(0, batch_idxs):

                img_batch, label_batch = self._load_batch(self.img_path, self.labels, idx)
                label_batch_shf = shuffle(label_batch)
                label_batch_shf2 = shuffle(label_batch)
                fake_img, input_label = self.sess.run([self.fake_img, self.input_label], feed_dict={self.real_img:img_batch, self.input_label:label_batch, self.input_label_shf:label_batch_shf})
                fake_img_o, input_label = self.sess.run([self.fake_img, self.input_label], feed_dict={self.real_img:img_batch, self.input_label:label_batch, self.input_label_shf:label_batch_shf2})
                for i in range(self.batch_size):
                    temp_fake = (fake_img[i]+1)*127.5
                    temp_fake_o = (fake_img_o[i]+1)*127.5
                    cv2.imwrite('./test/fake_e'+str(epoch)+str(idx)+str(i)+'_m.bmp', temp_fake)
                    cv2.imwrite('./test/fake_e'+str(epoch)+str(idx)+str(i)+'_o.bmp', temp_fake_o)
                    print(input_label[i])

            #for idx in range(0, batch_idxs):

            #    img_batch, label_batch = self._load_batch(self.img_path, self.labels, idx)
            #    modi_att = []
            #    label_batch_shf = shuffle(label_batch)

            #    fake_img, input_label = self.sess.run([self.fake_img, self.input_label], feed_dict={self.real_img:img_batch, self.input_label:label_batch, self.input_label_shf:label_batch_shf})
            #    for i in range(self.batch_size):
            #        temp_fake = (fake_img[i]+1)*127.5
            #        cv2.imwrite('./test/fake_e'+str(epoch)+str(idx)+str(i)+att_names[modi_att[i]]+'.bmp', temp_fake)
            #        print(input_label[i])


    def gradient_penalty(self):
        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        differences = self.fake_img - self.real_img # self.Y : 
        interpolates = self.real_img + (alpha * differences)
        gradients = tf.gradients(self.d_network(interpolates, reuse=True), [interpolates])[0]
        #red_idx = range(1, interpolates.shape.ndims)
        slopes = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2,3]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        return gradient_penalty