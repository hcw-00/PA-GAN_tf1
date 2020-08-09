from __future__ import division
import tensorflow as tf
from ops import *
from utils import *
import os
import functools
import tensorflow_graphics as tfg

#def conv_bn_relu(input, out_dim, k, s, activation_fn = tf.nn.relu):
#    net = slim.conv2d(input, out_dim, k, s)
#    net = slim.batch_norm(net, activation_fn=activation_fn)
#    return net
#def deconv_bn_relu(input, out_dim, k, s, activation_fn = tf.nn.relu):
#    net = slim.conv2d_transpose(input, out_dim, k, s)
#    net = slim.batch_norm(net, activation_fn=activation_fn)
#    return net
#def conv_ln_lrelu(input, out_dim, k, s, activation_fn = tf.nn.leaky_relu):
#    net = slim.conv2d(input, out_dim, k, s)
#    net = slim.layer_norm(net, activation_fn=activation_fn)
#    return net

def conv_set(input, out_dim, k, s, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu, scope="conv"):
    return slim.conv2d(input, out_dim, k, s, normalizer_fn=normalizer_fn, activation_fn=activation_fn, scope=scope)
def deconv_set(input, out_dim, k, s, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu, scope="deconv"):
    return slim.conv2d_transpose(input, out_dim, k, s, normalizer_fn=normalizer_fn, activation_fn=activation_fn, scope=scope)

conv_bn_relu = conv_set
conv_ln_lrelu = functools.partial(conv_set, normalizer_fn=slim.layer_norm, activation_fn=tf.nn.leaky_relu)
deconv_bn_relu = deconv_set


class D_network:
    def __call__(self, inputs, reuse=False):
        with tf.variable_scope("discriminator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(0.01)):
                net = self.d_common(inputs, reuse=reuse)
                #d_out = self.discriminator(net, reuse=reuse)
                #c_out = self.attribute_classifier(net, reuse=reuse)
                d_ = slim.fully_connected(net, 512, activation_fn=tf.nn.leaky_relu, scope='f_d')
                d_out = slim.fully_connected(d_, 1, reuse=reuse, scope='f_d_out')
                c_ = slim.fully_connected(net, 512, activation_fn=tf.nn.leaky_relu, scope='f_c')
                c_out = slim.fully_connected(c_, 13, reuse=reuse, scope='f_c_out')

        return d_out, c_out

    def d_common(self, inputs, reuse=False, name="d_common"):
        with tf.variable_scope(name, reuse=reuse):
            net = conv_ln_lrelu(inputs,64,4,2, scope="convl1")
            net = conv_ln_lrelu(net,128,4,2, scope="convl2")
            net = conv_ln_lrelu(net,256,4,2, scope="convl3")
            net = conv_ln_lrelu(net,512,4,2, scope="convl4")
            #net = conv_ln_lrelu(net,1024,4,2)
            net = slim.flatten(net)
        return net


class G_network:
    def __call__(self, input, b, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(0.01)):
                if reuse:
                    tf.get_variable_scope().reuse_variables()
                else:
                    assert tf.get_variable_scope().reuse is False                
                
                #en_out = self.encoder(input)
                #b = tf.to_float(b)
                # Downscale
                fa1 = conv_bn_relu(input, 64, 4, 2, scope="conv1")     # (?, 64, 64, 128)
                fa2 = conv_bn_relu(fa1, 128, 4, 2, scope="conv2")     # (?, 32, 32, 256)
                fa3 = conv_bn_relu(fa2, 256, 4, 2, scope="conv3")     # (?, 16, 16, 512)
                fb4 = conv_bn_relu(fa3, 512, 4, 2, scope="conv4")     # (?, 16, 16, 512)
                
                # Upscale
                #m_multi4 = tf.zeros_like(fb4[:,:,:,:13])     # (?, 16, 16, 13) <=== 0~255에서 zeros인지 아니면 -1~1 에서 zeros인지 check
                m_multi4 = None
                fb3, mb3, m_multi3 = self.attentive_editor(fa3, fb4, k=3, b_att=b, mask_in=m_multi4, reuse=False) #(out) fb3:,mb3, (in) fa[0]:32, fb4:16, mb4:16
                fb2, mb2, m_multi2 = self.attentive_editor(fa2, fb3, k=2, b_att=b, mask_in=m_multi3, reuse=False)
                fb1, mb1, m_multi1 = self.attentive_editor(fa1, fb2, k=1, b_att=b, mask_in=m_multi2, reuse=False)
                fb0, mb0, m_multi0 = self.attentive_editor(input, fb1, k=0, b_att=b, mask_in=m_multi1, reuse=False)

        #return fb1, [mb3,mb2,mb1], [m_multi3,m_multi2,m_multi1], [fa_in3, fa_in2, fa_in1], [ek_3, ek_2, ek_1], delta_m, b
        return fb0, [mb3,mb2,mb1], [m_multi3,m_multi2,m_multi1] #, [fa_in3, fa_in2, fa_in1]#, [ek_3, ek_2, ek_1], delta_m, b

    def attentive_editor(self, fa_in, fb_in, k, b_att, mask_in, reuse=False, name="attentive_editor"): #(out) fb3:,mb3, (in) fa3:32, fb4:16, mb4:16
        with tf.variable_scope(name+str(k), reuse=tf.AUTO_REUSE):
            #if reuse:
            #    tf.get_variable_scope().reuse_variables()

            #else:
            #    assert tf.get_variable_scope().reuse is False
            
            ## Gek ###
            ek = self.G_e_k(fb_in, b_att, k) # attribute feature
            ##########
            ## Gmk ###
            if mask_in is not None:
                #size = [mask_in.shape[1] * 2, mask_in.shape[2] * 2]
                #mask_in = tf.image.resize(mask_in, size)
                shape = [None, mask_in.shape[1] * 2, mask_in.shape[2] * 2, mask_in.shape[3]]
                mask_in = tfg.image.pyramid.upsample(mask_in, 1)[-1]
                mask_in.set_shape(shape)
            #mask_in = tf.image.resize_bicubic(mask_in, size)
            delta_mask_k = self.G_m_k(fa_in,ek,mask_in,b_att,k) # mask
            ##########
            if mask_in is not None:
                mbk = delta_mask_k + mask_in ## <=== check (is not None)
            else:
                mbk = delta_mask_k
            
            b = tf.reshape(tf.abs(tf.sign(b_att)), [-1, 1, 1, b_att.shape[-1]])
            m = tf.clip_by_value(tf.reduce_sum(b * tf.nn.sigmoid(mbk), axis=-1, keep_dims=True), 0.0, 1.0)
            #m = tf.clip_by_value(tf.reduce_mean(b * tf.nn.sigmoid(mbk), axis=-1, keep_dims=True), 0.0, 1.0)
            fb = (1-m)*fa_in + m*ek
        return fb, m, mbk #, fa_in #, ek, delta_mask_k, b

    def G_e_k(self, inputs, b, k, reuse=False, name="G_e_k"):

        with tf.variable_scope(name+str(k), reuse=tf.AUTO_REUSE):
            #if reuse:
            #    tf.get_variable_scope().reuse_variables()
            #else:
            #    assert tf.get_variable_scope().reuse is False
            net = tile_concat(inputs, b)
            if k!=0:
                net = deconv_bn_relu(net,64*2**(k-1),3,1, scope="deconv1")
                net = deconv_bn_relu(net,64*2**(k-1),3,2, scope="deconv2")
            else:
                net = deconv_bn_relu(net,32,3,1, scope="deconv1")
                net = slim.conv2d_transpose(net, 3, 3, 2, activation_fn=tf.nn.tanh, scope="deconv2")
        return net

    def G_m_k(self, fa_in, ek, mask_in, b, k, reuse=False, name="G_m_k"): # 32,32,32
        with tf.variable_scope(name+str(k), reuse=tf.AUTO_REUSE):
            #if reuse:
            #    tf.get_variable_scope().reuse_variables()
            #else:
            #    assert tf.get_variable_scope().reuse is False

            rm_none = lambda l: [x for x in l if x is not None]
            temp_ = rm_none([fa_in, ek, mask_in])
            net_concat = tile_concat(temp_, b)

            net_a = conv_bn_relu(net_concat,64*2**(k-1),1,1, scope="conv_a")

            net_b = conv_bn_relu(net_concat,64*2**(k-1),3,1, scope="conv_b")
            
            net_c = conv_bn_relu(net_concat,64*2**(k-1),3,1, scope="conv_c")
            net_c = conv_bn_relu(net_c,64*2**(k-1),3,1, scope="conv_c_2")
            
            net_d = conv_bn_relu(net_concat,64*2**(k-1),3,1, scope="conv_d")
            net_d = conv_bn_relu(net_d,64*2**(k-1),3,1, scope="conv_d_2")
            net_d = conv_bn_relu(net_d,64*2**(k-1),3,1, scope="conv_d_3")

            net = tf.concat([net_a, net_b, net_c, net_d], axis=-1)

            net = conv_bn_relu(net,64*2**(k),4,2, scope="conv_concat")
            delta_mask = slim.conv2d_transpose(net, b.shape[-1], 4, 2)

        return delta_mask


######################
# EVALUATION NETWORK #
######################
#def attiribute_predictor(inputs, reuse=False, name="attiribute_predictor"):

#    with tf.variable_scope(name):

#        if reuse:
#            tf.get_variable_scope().reuse_variables()
#        else:
#            assert tf.get_variable_scope().reuse is False


#        net = conv_bn_relu(input, 16, 3, 1)
#        net = conv_bn_relu(net, 16, 3, 1)
#        net = slim.pool(net,2,stride=2)

#        net = conv_bn_relu(net, 32, 3, 1)
#        net = conv_bn_relu(net, 32, 3, 1)
#        net = slim.pool(net,2,stride=2)

#        net = conv_bn_relu(net, 64, 3, 1)
#        net = conv_bn_relu(net, 64, 3, 1)
#        net = slim.pool(net,2,stride=2)

#        net = conv_bn_relu(net, 128, 3, 1)
#        net = conv_bn_relu(net, 128, 3, 1)
#        net = slim.pool(net,2,stride=2)


#        net = slim.flatten(net)
#        net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu)
#        net = slim.fully_connected(net, 13, activation_fn=tf.sigmoid)

#    return net



def tile_concat(a_list, b_list=[]): # this code from : https://github.com/LynnHo/PA-GAN-Tensorflow/blob/master/utils.py
    # tile all elements of `b_list` and then concat `a_list + b_list` along the channel axis
    # `a` shape: (N, H, W, C_a)
    # `b` shape: can be (N, 1, 1, C_b) or (N, C_b)
    a_list = list(a_list) if isinstance(a_list, (list, tuple)) else [a_list]
    b_list = list(b_list) if isinstance(b_list, (list, tuple)) else [b_list]
    for i, b in enumerate(b_list):
        b = tf.reshape(b, [-1, 1, 1, b.shape[-1]])
        b = tf.tile(b, [1, a_list[0].shape[1], a_list[0].shape[2], 1])
        b_list[i] = b
    return tf.concat(a_list + b_list, axis=-1)



def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target): # mae??? not mse??
    return tf.reduce_mean(tf.abs(in_-target))

def mse_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

def binary_criterion(logits, labels):
    return -tf.reduce_sum(labels*tf.log(logits)+(1-labels)*tf.log(1-logits))

def wgan_criterion(logit_fake, logit_real):
    return tf.reduce_mean(logit_fake) - tf.reduce_mean(logit_real)




def overlap_loss_fn(ms_multi, att_names):
    # ======================================
    # =        customized relation         =
    # ======================================

    full_overlap_pairs = [
        # ('Black_Hair', 'Blond_Hair'),
        # ('Black_Hair', 'Brown_Hair'),

        # ('Blond_Hair', 'Brown_Hair')
    ]

    non_overlap_pairs = [
        # ('Bald', 'Bushy_Eyebrows'),
        # ('Bald', 'Eyeglasses'),
        ('Bald', 'Mouth_Slightly_Open'),
        ('Bald', 'Mustache'),
        ('Bald', 'No_Beard'),

        ('Bangs', 'Mouth_Slightly_Open'),
        ('Bangs', 'Mustache'),
        ('Bangs', 'No_Beard'),

        ('Black_Hair', 'Mouth_Slightly_Open'),
        ('Black_Hair', 'Mustache'),
        ('Black_Hair', 'No_Beard'),

        ('Blond_Hair', 'Mouth_Slightly_Open'),
        ('Blond_Hair', 'Mustache'),
        ('Blond_Hair', 'No_Beard'),

        ('Brown_Hair', 'Mouth_Slightly_Open'),
        ('Brown_Hair', 'Mustache'),
        ('Brown_Hair', 'No_Beard'),

        # ('Bushy_Eyebrows', 'Mouth_Slightly_Open'),
        ('Bushy_Eyebrows', 'Mustache'),
        ('Bushy_Eyebrows', 'No_Beard'),

        # ('Eyeglasses', 'Mouth_Slightly_Open'),
        ('Eyeglasses', 'Mustache'),
        ('Eyeglasses', 'No_Beard'),
    ]

    # ======================================
    # =                 losses             =
    # ======================================

    full_overlap_pair_loss = tf.constant(0.0)
    for p in full_overlap_pairs:
        id1 = att_names.index(p[0])
        id2 = att_names.index(p[1])
        for m in ms_multi[-1:]:
            m1 = m[..., id1]
            m2 = m[..., id2]
            full_overlap_pair_loss += tf.losses.absolute_difference(m1, m2)

    non_overlap_pair_loss = tf.constant(0.0)
    for p in non_overlap_pairs:
        id1 = att_names.index(p[0])
        id2 = att_names.index(p[1])
        for m in ms_multi[-1:]:
            m1 = m[..., id1]
            m2 = m[..., id2]
            non_overlap_pair_loss += tf.reduce_mean(tf.nn.sigmoid(m1) * tf.nn.sigmoid(m2))

    return full_overlap_pair_loss, non_overlap_pair_loss

def get_random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop