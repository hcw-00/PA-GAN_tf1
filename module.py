from __future__ import division
import tensorflow as tf
from ops import *
from utils import *


def conv_bn_activ(input, out_dim, k, s, activation_fn = tf.nn.relu):
    net = slim.conv2d(input, out_dim, k, s)
    net = slim.batch_norm(net, activation_fn=activation_fn)
    return net
def deconv_bn_activ(input, out_dim, k, s, activation_fn = tf.nn.relu):
    #net = slim.deconv2d(input, out_dim, k, s)
    net = slim.conv2d_transpose(input, out_dim, k, s)
    net = slim.batch_norm(net, activation_fn=activation_fn)
    return net
def conv_ln_activ(input, out_dim, k, s, activation_fn = tf.nn.leaky_relu):
    net = slim.conv2d(input, out_dim, k, s)
    net = slim.layer_norm(net, activation_fn=activation_fn)
    return net



class D_network:
    def __call__(self, inputs, reuse=False):
        net = self.d_common(inputs, reuse=reuse)
        d_out = self.discriminator(net, reuse=reuse)
        c_out = self.attribute_classifier(net, reuse=reuse)
        return d_out, c_out

    def d_common(self, inputs, reuse=False, name="common"):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            net = conv_ln_activ(inputs,64,4,2)
            net = conv_ln_activ(net,128,4,2)
            net = conv_ln_activ(net,256,4,2)
            net = conv_ln_activ(net,512,4,2)
            #net = conv_ln_activ(net,1024,4,2)
            net = slim.flatten(net)
        return net

    def discriminator(self, inputs, reuse=False, name="discriminator"):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            net = slim.fully_connected(inputs, 512, activation_fn=tf.nn.leaky_relu)
            net = slim.fully_connected(net, 1)
        return net

    def attribute_classifier(self, inputs, reuse=False, name="attribute_classifier"):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            net = slim.fully_connected(inputs, 512, activation_fn=tf.nn.leaky_relu)
            net = slim.fully_connected(net, 13) #, activation_fn=tf.sigmoid) <-- tf.nn.sigmoid_cross_entropy_with_logits 에서 sigmoid 적용
        return net


class G_network:
    def __call__(self, xa, b, reuse=False):
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            #en_out = self.encoder(xa)

            fa2 = conv_bn_activ(xa, 64, 4, 2)     # (?, 64, 64, 128)
            fa3 = conv_bn_activ(fa2, 128, 4, 2)     # (?, 32, 32, 256)
            fb4 = conv_bn_activ(fa3, 256, 4, 2)     # (?, 16, 16, 512)

            m_multi4 = tf.zeros_like(fb4[:,:,:,:13])     # (?, 16, 16, 13)
            fb3, mb3, m_multi3 = self.attentive_editor(fa3, fb4, k=3, b_att=b, mask_in=m_multi4, reuse=False) #(out) fb3:,mb3, (in) fa[0]:32, fb4:16, mb4:16
            fb2, mb2, m_multi2 = self.attentive_editor(fa2, fb3, k=2, b_att=b, mask_in=m_multi3, reuse=False)
            fb1, mb1, m_multi1 = self.attentive_editor(xa, fb2, k=1, b_att=b, mask_in=m_multi2, reuse=False)

        return fb1, [mb3,mb2,mb1], [m_multi3,m_multi2,m_multi1]

    def attentive_editor(self, fa_in, fb_in, k, b_att, mask_in, reuse=False, name="attentive_editor"): #(out) fb3:,mb3, (in) fa3:32, fb4:16, mb4:16
        with tf.variable_scope(name+str(k)):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            
            ## Gek ###
            ek = self.attentive_editor_e_k(fb_in, b_att, k) # attribute feature
            ##########
            ## Gmk ###
            size = (mask_in.shape[1] * 2, mask_in.shape[2] * 2)
            mask_in = tf.image.resize(mask_in, size)
            delta_mask_k = self.attentive_editor_m_k(fa_in,ek,mask_in,b_att,k) # mask
            ##########

            mbk = delta_mask_k + mask_in
            
            b = tf.reshape(tf.abs(tf.sign(b_att)), [-1, 1, 1, b_att.shape[-1]])
            m = tf.clip_by_value(tf.reduce_sum(b * tf.nn.sigmoid(mbk), axis=-1, keep_dims=True), 0.0, 1.0)
            print("test")
            fb_k = (1-m)*fa_in + m*ek
        return fb_k, m, mbk

    def attentive_editor_e_k(self, inputs, b, k, reuse=False, name="attentive_editor_e_k"):

        with tf.variable_scope(name):

            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            if k!=1:
                net = tile_concat(inputs, b)
                net = deconv_bn_activ(net,32*2**(k-1),3,1)
                net = deconv_bn_activ(net,32*2**(k-1),3,2)
            else:
                net = tile_concat(inputs, b)
                net = deconv_bn_activ(net,16,3,1)
                net = slim.conv2d_transpose(net, 3, 3, 2, activation_fn=tf.nn.tanh)
        return net

    def attentive_editor_m_k(self, fa_in, ek, mask_in, b, k, reuse=False, name="attentive_editor_m_k"): # 32,32,32
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            rm_none = lambda l: [x for x in l if x is not None]
            temp_ = rm_none([fa_in, ek, mask_in])
            net_concat = tile_concat(temp_, b)

            net_a = conv_bn_activ(net_concat,32*2**(k-1),1,1)

            net_b = conv_bn_activ(net_concat,32*2**(k-1),3,1)
            
            net_c = conv_bn_activ(net_concat,32*2**(k-1),3,1)
            net_c = conv_bn_activ(net_c,32*2**(k-1),3,1)
            
            net_d = conv_bn_activ(net_concat,32*2**(k-1),3,1)
            net_d = conv_bn_activ(net_d,32*2**(k-1),3,1)
            net_d = conv_bn_activ(net_d,32*2**(k-1),3,1)

            net = tf.concat([net_a, net_b, net_c, net_d], axis=-1)

            net = conv_bn_activ(net,32*2**(k),4,2)
            net = slim.conv2d_transpose(net, b.shape[-1], 4, 2)

        return net




######################
# EVALUATION NETWORK #
######################
def attiribute_predictor(inputs, reuse=False, name="attiribute_predictor"):

    with tf.variable_scope(name):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False


        net = conv_bn_activ(input, 16, 3, 1)
        net = conv_bn_activ(net, 16, 3, 1)
        net = slim.pool(net,2,stride=2)

        net = conv_bn_activ(net, 32, 3, 1)
        net = conv_bn_activ(net, 32, 3, 1)
        net = slim.pool(net,2,stride=2)

        net = conv_bn_activ(net, 64, 3, 1)
        net = conv_bn_activ(net, 64, 3, 1)
        net = slim.pool(net,2,stride=2)

        net = conv_bn_activ(net, 128, 3, 1)
        net = conv_bn_activ(net, 128, 3, 1)
        net = slim.pool(net,2,stride=2)


        net = slim.flatten(net)
        net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu)
        net = slim.fully_connected(net, 13, activation_fn=tf.sigmoid)

    return net



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

def gradient_penalty():
    alpha = tf.random_uniform(shape=[self.flags.batch_size, 1, 1, 1], minval=0., maxval=1.)
    differences = self.g_samples - self.Y
    interpolates = self.Y + (alpha * differences)
    gradients = tf.gradients(self.discriminator(interpolates, is_reuse=True), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    return gradient_penalty


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