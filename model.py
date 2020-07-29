from __future__ import division
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
from imgaug import augmenters as iaa

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
        img_name = np.genfromtxt(dir_path + "test_label.txt", dtype=str, usecols=0)
        img_path = [img_dir_path + i for i in img_name]
        labels = np.genfromtxt(dir_path + "test_label.txt", dtype=int, usecols=range(1,41))
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
            img_batch.append(img)
            temp_label = [int(j) for j in labels[i+idx*self.batch_size]]
            label = (temp_label+np.ones_like(temp_label))//2
            label_batch.append(label)
        return img_batch, label_batch


    def _build_model(self, args):

        self.real_img = tf.placeholder(tf.float32, [None,128,128,3], name='input')
        self.input_label = tf.placeholder(tf.float32, [None,13], name='label')
        

        ##
        g_network = G_network()
        d_network = D_network()
        
        self.fake_img, mask, ms_multi = g_network(self.real_img, self.input_label)
        d_real, c_real = d_network(self.real_img, reuse=False)
        d_fake, c_fake = d_network(self.fake_img, reuse=True)


        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.c_vars = [var for var in t_vars if 'attribute_classifier' in var.name]
        print("trainable variables : ")
        print(t_vars)
        
        # losses
        self.l_att = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_label,logits=c_fake))#binary_criterion(labels=self.input_label,logits=c_fake)
        self.l_c = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_label,logits=c_real))#binary_criterion(labels=self.input_label, logits=c_real)
        self.l_adv_g = -tf.reduce_mean(d_fake)
        self.l_adv_d = self.criterionWGAN(d_fake, d_real) # + GP
        self.l_spa = tf.reduce_sum([tf.reduce_mean(m) * w for m, w in zip(mask, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])])
        self.l_full_ovl, self.l_none_ovl = overlap_loss_fn(ms_multi, self.att_names)

        self.g_loss = 20*self.l_att + \
            self.l_adv_g + 0.05*self.l_spa + \
            self.l_full_ovl + \
            self.l_none_ovl #+ regularization_loss
        self.d_loss = self.l_c + self.l_adv_d

        self.loss_summary = tf.summary.scalar("loss", self.l_c)
        

    def train(self, args):
        
        #self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.lr = args.lr
        
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.lr, global_step, args.epoch_step, 1, staircase=False)

        self.D_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1, beta2=args.beta2) \
            .minimize(self.d_loss, var_list=[self.d_vars, self.c_vars], global_step = global_step)
        self.G_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1, beta2=args.beta2) \
            .minimize(self.g_loss, var_list=[self.g_vars], global_step = global_step)
        
        print("initialize")
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()


        for epoch in range(args.epoch):
            
            batch_idxs = len(self.labels) // self.batch_size

            self.img_path, self.labels = shuffle(self.img_path, self.labels)
            
            for idx in range(0, batch_idxs):

                img_batch, label_batch = self._load_batch(self.img_path, self.labels, idx)
                #img_batch = np.zeros((self.batch_size,256,256,3))
                _, d_loss,l_c,l_adv_d = self.sess.run([self.D_optim, self.d_loss,self.l_c,self.l_adv_d], feed_dict={self.real_img:img_batch, self.input_label:label_batch})

                fake_img, _, g_loss,l_att,l_adv_g,l_spa,l_full_ovl,l_none_ovl = self.sess.run([self.fake_img,self.G_optim, self.g_loss,self.l_att,self.l_adv_g,self.l_spa,self.l_full_ovl,self.l_none_ovl], feed_dict={self.real_img:img_batch, self.input_label:label_batch})
                #self.writer.add_summary(summary_str, counter)

                #counter += 1
                if idx%1==0:
                    print(("Epoch: [%2d] [%4d/%4d] | D adv loss: %4.4f | G adv loss: %4.4f | time: %4.2f" % (
                        epoch, idx, batch_idxs, d_loss, g_loss, time.time() - start_time)))
                    #print(("Epoch: [%2d] [%4d/%4d] | D adv loss: %4.4f, %4.4f | G adv loss: %4.4f,%4.4f,%4.4f,%4.4f,%4.4f | time: %4.2f" % (
                    #    epoch, idx, batch_idxs, l_c, l_adv_d, 20*l_att,l_adv_g,0.05*l_spa,l_full_ovl,l_none_ovl, time.time() - start_time)))

                if idx%20 == 0:
                    #self.save(args.checkpoint_dir, counter)
                    temp_fake = (fake_img[0]+1)*127.5
                    #temp_recon = np.reshape(input_batch[j]*255, (28,28))
                    cv2.imwrite('./sample/fake_e'+str(epoch)+str(idx)+'.bmp', temp_fake)
                    #cv2.imwrite('./test/recon_'+str(j)+'.bmp', temp_image)
                if epoch == args.epoch-1 and idx == batch_idxs-1:
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
            #temp_ckpt = 'dnn.model-23401'
            #ckpt_name = os.path.basename(temp_ckpt)
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
        self.test_image, self.test_label = shuffle(self.test_image, self.test_label)
        for epoch in range(1):
            
            batch_idxs = len(self.test_label) // self.batch_size

            latent_list = []
            
            for idx in range(1):

                input_batch = self._load_batch(idx, self.test_image)

                z_input = np.random.normal(0,1,[self.batch_size,128])

                recon_img, w_latent = self.sess.run([self.recon_image, self.w_test], feed_dict={self.z_input : z_input, self.real_input : input_batch})

                for j in range(4):
                    temp_image = np.reshape(recon_img[j]*255, (28,28))
                    temp_real = np.reshape(input_batch[j]*255, (28,28))
                    cv2.imwrite('./test/'+str(j)+'_real.bmp', temp_real)
                    cv2.imwrite('./test/'+str(j)+'_recon.bmp', temp_image)
                    latent_list.append(w_latent[j])
                counter += 1

        latent_step = (latent_list[3] - latent_list[2])/6
        latent_concat = np.zeros([28, 28*7])
        for i in range(7):
            #self.gen_from_latent = self.generator(self.latent, eta, reuse=True, name='generator')
            temp_latent = latent_list[2] + i*latent_step
            temp_latent = np.expand_dims(temp_latent, axis=0)
            temp_img = self.sess.run([self.gen_from_latent], feed_dict={self.latent: temp_latent})
            latent_concat[:,i*28:(i+1)*28] = np.reshape(temp_img, (28,28))*255

        cv2.imwrite('./test/latent_walk.bmp', latent_concat)

        #self.gen_from_latent = self.generator(self.latent, eta, reuse=True, name='generator')
        #for i in range(100):
        #    for j in range(100)

 