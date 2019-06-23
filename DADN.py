# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import os
import shutil
from PIL import Image
import time
import random
import sys
import matlab.engine

import input_util
from mmd import mix_rbf_mmd2
from model import *

# Parameter setting
tf.app.flags.DEFINE_float('unseenPer', 0.5, "unseenPer")
tf.app.flags.DEFINE_string('GPU', '0',"GPU to use")
tf.app.flags.DEFINE_string('dataset', 'xmedianet',"dataset to use")
tf.app.flags.DEFINE_boolean('zeroShot', True, "zeroShot")
tf.app.flags.DEFINE_float('train_dropout', 0.9, "train_dropout")
tf.app.flags.DEFINE_integer('batch_size', 64, "batch_size")
tf.app.flags.DEFINE_float('learn_rate', 0.0001, "learn_rate")
tf.app.flags.DEFINE_float('decay_rate', 0.96, "decay_rate")
tf.app.flags.DEFINE_integer('decay_steps', 10, "decay_steps")
tf.app.flags.DEFINE_integer('total_epoch', 10, "total_epoch")
tf.app.flags.DEFINE_float('margin', 20, "margin")
tf.app.flags.DEFINE_boolean('to_restore', False, "to_restore")
tf.app.flags.DEFINE_boolean('to_save', True, "to_save")
tf.app.flags.DEFINE_string('data_dir', '/data1/chijingze/DADN/data/',"data dir")
tf.app.flags.DEFINE_string('check_dir', '/data1/chijingze/DADN/check/',"check dir")
tf.app.flags.DEFINE_integer('input_img_fea_dim', 4096, "input_img_fea_dim")
tf.app.flags.DEFINE_integer('input_txt_fea_dim', 300, "input_txt_fea_dim")
tf.app.flags.DEFINE_integer('input_cate_dim', 300, "input_cate_dim")

FLAGS = tf.app.flags.FLAGS
unseenPer = FLAGS.unseenPer
dataset = FLAGS.dataset
zeroShot = FLAGS.zeroShot
train_dropout = FLAGS.train_dropout
batch_size = FLAGS.batch_size
learn_rate = FLAGS.learn_rate
decay_rate = FLAGS.decay_rate
decay_steps = FLAGS.decay_steps
total_epoch = FLAGS.total_epoch
margin = FLAGS.margin
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
to_restore = FLAGS.to_restore
to_save = FLAGS.to_save

input_img_fea_dim = FLAGS.input_img_fea_dim
input_txt_fea_dim = FLAGS.input_txt_fea_dim
input_cate_dim = FLAGS.input_cate_dim

data_dir = FLAGS.data_dir+dataset+'/'
check_dir = FLAGS.check_dir+dataset+'/'
savePath = check_dir+"DADN_"+dataset+"_"+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+"_"

eng = matlab.engine.start_matlab()

class DADN():

    def input_setup_train(self):

        self.image_input_train,self.text_input_train,self.label_train,self.cateFeaAll,self.misCateFeaAll,self.train_num,_ = \
        input_util.read_feature_from_txt_pair(data_dir+"img_train_fea.txt", data_dir+"img_train_list.txt",data_dir+"txt_train_fea.txt",\
            data_dir+"txt_train_list.txt",data_dir+"cate_fea.txt",zeroShot,dataset,unseenPer)  


    def input_setup_test(self):

        self.image_input_test_query,self.text_input_test_query,self.label_test_query,self.cateFeaAll_test_query,_,self.test_query_num,_ = \
        input_util.read_feature_from_txt_pair(data_dir+"img_test_fea.txt", data_dir+"img_test_list.txt", \
        	data_dir+"txt_test_fea.txt", data_dir+"txt_test_list.txt",data_dir+"cate_fea.txt",False,dataset,unseenPer)  

        self.image_input_test_data,self.text_input_test_data,self.label_test_data,self.cateFeaAll_test_data,_,self.test_data_num,_ = \
        input_util.read_feature_from_txt_pair(data_dir+"img_train_fea.txt", data_dir+"img_train_list.txt", \
        	data_dir+"txt_train_fea.txt", data_dir+"txt_train_list.txt",data_dir+"cate_fea.txt",False,dataset,unseenPer)    

    
    def input_read_train(self, sess):

        self.input_img = np.zeros((batch_size, input_img_fea_dim))
        self.input_txt = np.zeros((batch_size, input_txt_fea_dim))
        self.input_cate = np.zeros((batch_size, input_cate_dim))
        self.input_mis_img = np.zeros((batch_size, input_img_fea_dim))
        self.input_mis_txt = np.zeros((batch_size, input_txt_fea_dim))

        self.input_img,self.input_txt,self.input_cate,self.input_mis_img,self.input_mis_txt = \
        input_util.get_batch_shuffle_pair_mis(self.image_input_train,self.text_input_train,self.cateFeaAll,self.train_num,batch_size)

    def model_setup(self,for_train):

        self.input_GAN_img = tf.placeholder(tf.float32, [batch_size, input_img_fea_dim], name="input_GAN_img")
        self.input_GAN_txt = tf.placeholder(tf.float32, [batch_size, input_txt_fea_dim], name="input_GAN_txt")
        self.input_cate_fea = tf.placeholder(tf.float32, [batch_size, input_cate_dim], name="input_cate_fea")
        self.input_GAN_mis_img = tf.placeholder(tf.float32, [batch_size, input_img_fea_dim], name="input_GAN_mis_img")
        self.input_GAN_mis_txt = tf.placeholder(tf.float32, [batch_size, input_txt_fea_dim], name="input_GAN_mis_txt")
        self.keep_prob = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.lr = tf.placeholder(tf.float32, shape=[], name="lr")

        with tf.variable_scope("Model") as scope:
            # SeGAN for img
            self.fake_cate_fea_img = SeGAN_img(self.input_GAN_img,self.keep_prob,for_train,"g_A_img")           
            self.rec_fake_cate_fea_img = SeGAN_discriminator_img_MMD(self.fake_cate_fea_img,self.input_GAN_img,1,for_train, "d_A_img")

            # ReGAN for img
            self.cyc_input_img = ReGAN_img(self.fake_cate_fea_img,self.keep_prob,for_train,"g_B_img")
            self.rec_cyc_input_img = ReGAN_discriminator_img_MMD(self.cyc_input_img,1, for_train,"d_B_img")

            # SeGAN for txt
            self.fake_cate_fea_txt = SeGAN_txt(self.input_GAN_txt,self.keep_prob,for_train,"g_A_txt")           
            self.rec_fake_cate_fea_txt = SeGAN_discriminator_txt_MMD(self.fake_cate_fea_txt,self.input_GAN_txt,1, for_train,"d_A_txt")

            # ReGAN for txt
            self.cyc_input_txt = ReGAN_txt(self.fake_cate_fea_txt,self.keep_prob,for_train,"g_B_txt")
            self.rec_cyc_input_txt = ReGAN_discriminator_txt_MMD(self.cyc_input_txt,1, for_train,"d_B_txt")

            scope.reuse_variables()
            self.mis_cate_fea_img = SeGAN_img(self.input_GAN_mis_img,self.keep_prob,for_train,"g_A_img")
            self.mis_cate_fea_txt = SeGAN_txt(self.input_GAN_mis_txt,self.keep_prob,for_train,"g_A_txt")

            self.rec_cate_fea_img = SeGAN_discriminator_img_MMD(self.input_cate_fea,self.input_GAN_img,1,for_train, "d_A_img")
            self.rec_cate_fea_txt = SeGAN_discriminator_txt_MMD(self.input_cate_fea,self.input_GAN_txt,1, for_train,"d_A_txt")

            self.rec_input_img = ReGAN_discriminator_img_MMD(self.input_GAN_img,1,for_train, "d_B_img")
            self.rec_input_txt = ReGAN_discriminator_txt_MMD(self.input_GAN_txt,1,for_train, "d_B_txt")

            self.rec_fake_cate_fea_img_txt = SeGAN_discriminator_img_MMD(self.fake_cate_fea_txt,self.input_GAN_img,1,for_train, "d_A_img")
            self.rec_fake_cate_fea_txt_img = SeGAN_discriminator_txt_MMD(self.fake_cate_fea_img,self.input_GAN_txt,1, for_train,"d_A_txt")
    

    def loss_calc(self):

        # For img
        d_loss_A_img = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=self.rec_cate_fea_img[-1],labels=tf.ones_like(self.rec_cate_fea_img[-1])) + \
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.rec_fake_cate_fea_img[-1],labels=tf.zeros_like(self.rec_fake_cate_fea_img[-1])) + \
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.rec_fake_cate_fea_img_txt[-1],labels=tf.zeros_like(self.rec_fake_cate_fea_img_txt[-1])) )
              
        d_loss_B_img = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.rec_cyc_input_img[-1],labels=tf.zeros_like(self.rec_cyc_input_img[-1])) + \
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.rec_input_img[-1],labels=tf.ones_like(self.rec_input_img[-1])) )

        bandwidths = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
        concat_rec_fake_cate_fea_img = tf.concat(self.rec_fake_cate_fea_img[0:-1],1)
        concat_rec_cate_fea_img = tf.concat(self.rec_cate_fea_img[0:-1],1)

        concat_rec_cyc_input_img = tf.concat(self.rec_cyc_input_img[0:-1],1)
        concat_rec_input_img = tf.concat(self.rec_input_img[0:-1],1)

        concat_rec_fake_cate_fea_img = [concat_rec_fake_cate_fea_img]
        concat_rec_cate_fea_img = [concat_rec_cate_fea_img]
        concat_rec_cyc_input_img = [concat_rec_cyc_input_img]
        concat_rec_input_img = [concat_rec_input_img]

        g_loss_A_img = tf.sqrt(mix_rbf_mmd2(concat_rec_fake_cate_fea_img[0],concat_rec_cate_fea_img[0], sigmas=bandwidths))
        g_loss_B_img = tf.sqrt(mix_rbf_mmd2(concat_rec_cyc_input_img[0],concat_rec_input_img[0], sigmas=bandwidths))


        concat_rec_fake_cate_fea_img_txt = tf.concat(self.rec_fake_cate_fea_img_txt[0:-1],1)
        concat_rec_cate_fea_img = tf.concat(self.rec_cate_fea_img[0:-1],1)

        concat_rec_fake_cate_fea_img_txt = [concat_rec_fake_cate_fea_img_txt]
        concat_rec_cate_fea_img = [concat_rec_cate_fea_img]

        g_loss_A_img = g_loss_A_img + tf.sqrt(mix_rbf_mmd2(concat_rec_fake_cate_fea_img_txt[0],concat_rec_cate_fea_img[0], sigmas=bandwidths))

        true_dis_loss_img = tf.reduce_sum(tf.nn.l2_loss(self.fake_cate_fea_img-self.input_cate_fea))
        I_false_dis_loss_img = tf.reduce_sum(tf.nn.l2_loss(self.fake_cate_fea_img-self.mis_cate_fea_img))
        T_false_dis_loss_img = tf.reduce_sum(tf.nn.l2_loss(self.fake_cate_fea_img-self.mis_cate_fea_txt))

        I_re_dis_loss = tf.reduce_sum(tf.nn.l2_loss(self.input_GAN_img-self.cyc_input_img))

        g_loss_A_img = g_loss_A_img + tf.maximum(0.,margin*true_dis_loss_img-I_false_dis_loss_img-T_false_dis_loss_img) 
        g_loss_B_img = g_loss_B_img + I_re_dis_loss


        # For txt
        d_loss_A_txt = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.rec_cate_fea_txt[-1],labels=tf.ones_like(self.rec_cate_fea_txt[-1])) + \
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.rec_fake_cate_fea_txt[-1],labels=tf.zeros_like(self.rec_fake_cate_fea_txt[-1])) + \
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.rec_fake_cate_fea_txt_img[-1],labels=tf.zeros_like(self.rec_fake_cate_fea_txt_img[-1])))

        d_loss_B_txt = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.rec_cyc_input_txt[-1],labels=tf.zeros_like(self.rec_cyc_input_txt[-1])) + \
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.rec_input_txt[-1],labels=tf.ones_like(self.rec_input_txt[-1])))

        concat_rec_fake_cate_fea_txt = tf.concat(self.rec_fake_cate_fea_txt[0:-1],1)
        concat_rec_cate_fea_txt = tf.concat(self.rec_cate_fea_txt[0:-1],1)

        concat_rec_cyc_input_txt = tf.concat(self.rec_cyc_input_txt[0:-1],1)
        concat_rec_input_txt = tf.concat(self.rec_input_txt[0:-1],1)

        concat_rec_fake_cate_fea_txt = [concat_rec_fake_cate_fea_txt]
        concat_rec_cate_fea_txt = [concat_rec_cate_fea_txt]
        concat_rec_cyc_input_txt = [concat_rec_cyc_input_txt]
        concat_rec_input_txt = [concat_rec_input_txt]

        g_loss_A_txt = tf.sqrt(mix_rbf_mmd2(concat_rec_fake_cate_fea_txt[0],concat_rec_cate_fea_txt[0], sigmas=bandwidths))
        g_loss_B_txt = tf.sqrt(mix_rbf_mmd2(concat_rec_cyc_input_txt[0],concat_rec_input_txt[0], sigmas=bandwidths))

        concat_rec_fake_cate_fea_txt_img = tf.concat(self.rec_fake_cate_fea_txt_img[0:-1],1)
        concat_rec_cate_fea_txt = tf.concat(self.rec_cate_fea_txt[0:-1],1)

        concat_rec_fake_cate_fea_txt_img = [concat_rec_fake_cate_fea_txt_img]
        concat_rec_cate_fea_txt = [concat_rec_cate_fea_txt]

        g_loss_A_txt = g_loss_A_txt + tf.sqrt(mix_rbf_mmd2(concat_rec_fake_cate_fea_txt_img[0],concat_rec_cate_fea_txt[0], sigmas=bandwidths))


        true_dis_loss_txt = tf.reduce_sum(tf.nn.l2_loss(self.fake_cate_fea_txt-self.input_cate_fea))
        I_false_dis_loss_txt = tf.reduce_sum(tf.nn.l2_loss(self.fake_cate_fea_txt-self.mis_cate_fea_img))
        T_false_dis_loss_txt = tf.reduce_sum(tf.nn.l2_loss(self.fake_cate_fea_txt-self.mis_cate_fea_txt))

        T_re_dis_loss = tf.reduce_sum(tf.nn.l2_loss(self.input_GAN_txt-self.cyc_input_txt))

        g_loss_A_txt = g_loss_A_txt + tf.maximum(0.,margin*true_dis_loss_txt-I_false_dis_loss_txt-T_false_dis_loss_txt) 
        g_loss_B_txt = g_loss_B_txt + T_re_dis_loss


        # Total loss
        d_loss_A_sum = d_loss_A_img + d_loss_A_txt
        g_loss_A_sum = g_loss_A_img + g_loss_A_txt
        d_loss_B_sum = d_loss_B_img + d_loss_B_txt
        g_loss_B_sum = g_loss_B_img + g_loss_B_txt

        self.d_loss_A_sum_show = d_loss_A_sum
        self.g_loss_A_sum_show = g_loss_A_sum
        self.d_loss_B_sum_show = d_loss_B_sum
        self.g_loss_B_sum_show = g_loss_B_sum

        optimizer = tf.train.AdamOptimizer(self.lr)

        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'Model/d_A_img' in var.name or 'Model/d_A_txt' in var.name]
        g_A_vars = [var for var in self.model_vars if 'Model/g_A_img' in var.name or 'Model/g_A_txt' in var.name]
        d_B_vars = [var for var in self.model_vars if 'Model/d_B_img' in var.name or 'Model/d_B_txt' in var.name]
        g_B_vars = [var for var in self.model_vars if 'Model/g_B_img' in var.name or 'Model/g_B_txt' in var.name or 'Model/g_A_img' in var.name or 'Model/g_A_txt' in var.name]

        self.d_A_trainer = optimizer.minimize(d_loss_A_sum, var_list=d_A_vars)
        self.g_A_trainer = optimizer.minimize(g_loss_A_sum, var_list=g_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B_sum, var_list=d_B_vars)     
        self.g_B_trainer = optimizer.minimize(g_loss_B_sum, var_list=g_B_vars)

        print("All training Parameters:")
        print("d_A_vars:")
        for var in d_A_vars: print(var.name)
        print("g_A_vars:")
        for var in g_A_vars: print(var.name)
        print("d_B_vars:")
        for var in d_B_vars: print(var.name)
        print("g_B_vars:")
        for var in g_B_vars: print(var.name)

        self.g_A_loss_summ = tf.summary.scalar("g_A_loss_img", g_loss_A_sum)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss_img", d_loss_A_sum)
        self.g_B_loss_summ = tf.summary.scalar("g_A_loss_txt", g_loss_B_sum)
        self.d_B_loss_summ = tf.summary.scalar("d_A_loss_txt", d_loss_B_sum)
        
    def train(self):

        # Load Dataset from the dataset folder
        self.input_setup_train()  
        print("Train data load successfully, train num:"+str(self.train_num))
        self.input_setup_test()
        print("Test data load successfully")

        # Build the network
        self.model_setup(True)
        
        # Loss function
        self.loss_calc()

        # Initializing the global variables
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1) 

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)

            # Restore the model to run the model from last checkpoint
            if to_restore:
                chkpt_fname = tf.train.latest_checkpoint(check_dir)
                saver.restore(sess, chkpt_fname)

            if not os.path.exists(check_dir):
                os.makedirs(check_dir)

            curr_lr = learn_rate

            # Training Loop
            for epoch in range(sess.run(self.global_step),total_epoch):                
                print ("In the epoch ", epoch)
                if to_save:
                    saver.save(sess,os.path.join(check_dir,"DADN"),global_step=epoch)

                # learning rate for per the epoch number
                if(epoch < decay_steps) :
                    curr_lr = learn_rate
                else:
                    if epoch % decay_steps == 0:
                        curr_lr = curr_lr * decay_rate

                print("learn rate:",curr_lr)

                for ptr in range(0,self.train_num/batch_size):

                    self.input_read_train(sess)

                    _, summary_str,g_A_show = sess.run([self.g_A_trainer, self.g_A_loss_summ,self.g_loss_A_sum_show],\
                    feed_dict={self.input_GAN_img:self.input_img,self.input_GAN_txt:self.input_txt, self.input_cate_fea:self.input_cate,\
                    self.input_GAN_mis_img:self.input_mis_img,self.input_GAN_mis_txt:self.input_mis_txt, self.lr:curr_lr,self.keep_prob: train_dropout})
                    _, summary_str,g_A_show = sess.run([self.g_A_trainer, self.g_A_loss_summ,self.g_loss_A_sum_show],\
                    feed_dict={self.input_GAN_img:self.input_img,self.input_GAN_txt:self.input_txt, self.input_cate_fea:self.input_cate,\
                    self.input_GAN_mis_img:self.input_mis_img,self.input_GAN_mis_txt:self.input_mis_txt, self.lr:curr_lr,self.keep_prob: train_dropout})                   

                    _, summary_str,g_B_show = sess.run([self.g_B_trainer, self.g_B_loss_summ,self.g_loss_B_sum_show],\
                    feed_dict={self.input_GAN_img:self.input_img, self.input_GAN_txt:self.input_txt, self.lr:curr_lr,self.keep_prob: train_dropout})

                    _, summary_str,d_A_show = sess.run([self.d_A_trainer, self.d_A_loss_summ,self.d_loss_A_sum_show],\
                    feed_dict={self.input_GAN_img:self.input_img, self.input_GAN_txt:self.input_txt, self.input_cate_fea:self.input_cate, self.lr:curr_lr,self.keep_prob: train_dropout})                  

                    _, summary_str,d_B_show = sess.run([self.d_B_trainer, self.d_B_loss_summ,self.d_loss_B_sum_show],\
                    feed_dict={self.input_GAN_img:self.input_img, self.input_GAN_txt:self.input_txt, self.lr:curr_lr,self.keep_prob: train_dropout})     
                       
                sess.run(tf.assign(self.global_step, epoch + 1))

            # Test model
            self.test(sess,total_epoch,False)  


    def test(self,sess,epoch,output):

        print("Testing the results epoch:"+str(epoch))

        queryImageOutfile = savePath+str(epoch)+"_queryImageFea.txt"
        queryTextOutfile = savePath+str(epoch)+"_queryTextFea.txt"
        dataImageOutfile = savePath+str(epoch)+"_dataImageFea.txt"
        dataTextOutfile = savePath+str(epoch)+"_dataTextFea.txt"

        loopTimes = self.test_query_num / batch_size
        print("Query loopTimes:"+str(loopTimes))
        for i in range(0,loopTimes):
            self.input_img_query,self.input_txt_query,self.input_cate_query = \
            input_util.get_batch_seq_pair(self.image_input_test_query,self.text_input_test_query,self.cateFeaAll_test_query,self.test_query_num,i*batch_size,batch_size)
            fake_cate_img_temp,fake_cate_txt_temp,cyc_input_img_temp,cyc_input_txt_temp = sess.run([self.fake_cate_fea_img,self.fake_cate_fea_txt,self.cyc_input_img,self.cyc_input_txt],feed_dict= \
                {self.input_GAN_img:self.input_img_query,self.input_GAN_txt:self.input_txt_query,self.keep_prob: 1})

            if i == 0:
                fake_cate_img_list_query = fake_cate_img_temp
                fake_cate_txt_list_query = fake_cate_txt_temp
                cyc_input_img_list_query = cyc_input_img_temp
                cyc_input_txt_list_query = cyc_input_txt_temp
            else:
                fake_cate_img_list_query = np.append(fake_cate_img_list_query,fake_cate_img_temp,axis = 0)
                fake_cate_txt_list_query = np.append(fake_cate_txt_list_query,fake_cate_txt_temp,axis = 0)
                cyc_input_img_list_query = np.append(cyc_input_img_list_query,cyc_input_img_temp,axis = 0)
                cyc_input_txt_list_query = np.append(cyc_input_txt_list_query,cyc_input_txt_temp,axis = 0)

        remainNum = self.test_query_num - loopTimes * batch_size
        if remainNum > 0:
            self.input_img_query,self.input_txt_query,self.input_cate_query = \
            input_util.get_batch_seq_pair(self.image_input_test_query,self.text_input_test_query,self.cateFeaAll_test_query,self.test_query_num,self.test_query_num-batch_size,batch_size)
            fake_cate_img_temp,fake_cate_txt_temp,cyc_input_img_temp,cyc_input_txt_temp = sess.run([self.fake_cate_fea_img,self.fake_cate_fea_txt,self.cyc_input_img,self.cyc_input_txt],feed_dict= \
                {self.input_GAN_img:self.input_img_query,self.input_GAN_txt:self.input_txt_query,self.keep_prob: 1})
            fake_cate_img_list_query = np.append(fake_cate_img_list_query,fake_cate_img_temp[-remainNum:],axis = 0)
            fake_cate_txt_list_query = np.append(fake_cate_txt_list_query,fake_cate_txt_temp[-remainNum:],axis = 0)
            cyc_input_img_list_query = np.append(cyc_input_img_list_query,cyc_input_img_temp[-remainNum:],axis = 0)
            cyc_input_txt_list_query = np.append(cyc_input_txt_list_query,cyc_input_txt_temp[-remainNum:],axis = 0)


        loopTimes = self.test_data_num / batch_size
        print("Data loopTimes:"+str(loopTimes))
        for i in range(0,loopTimes):
            self.input_img_data,self.input_txt_data,self.input_cate_data = \
            input_util.get_batch_seq_pair(self.image_input_test_data,self.text_input_test_data,self.cateFeaAll_test_data,self.test_data_num,i*batch_size,batch_size)
            fake_cate_img_temp,fake_cate_txt_temp,cyc_input_img_temp,cyc_input_txt_temp = sess.run([self.fake_cate_fea_img,self.fake_cate_fea_txt,self.cyc_input_img,self.cyc_input_txt],feed_dict= \
                {self.input_GAN_img:self.input_img_data,self.input_GAN_txt:self.input_txt_data,self.keep_prob: 1})

            if i == 0:
                fake_cate_img_list_data = fake_cate_img_temp
                fake_cate_txt_list_data = fake_cate_txt_temp
                cyc_input_img_list_data = cyc_input_img_temp
                cyc_input_txt_list_data = cyc_input_txt_temp
            else:
                fake_cate_img_list_data = np.append(fake_cate_img_list_data,fake_cate_img_temp,axis = 0)
                fake_cate_txt_list_data = np.append(fake_cate_txt_list_data,fake_cate_txt_temp,axis = 0)
                cyc_input_img_list_data = np.append(cyc_input_img_list_data,cyc_input_img_temp,axis = 0)
                cyc_input_txt_list_data = np.append(cyc_input_txt_list_data,cyc_input_txt_temp,axis = 0)

        remainNum = self.test_data_num - loopTimes * batch_size
        if remainNum > 0:
            self.input_img_data,self.input_txt_data,self.input_cate_data = \
            input_util.get_batch_seq_pair(self.image_input_test_data,self.text_input_test_data,self.cateFeaAll_test_data,self.test_data_num,self.test_data_num-batch_size,batch_size)
            fake_cate_img_temp,fake_cate_txt_temp,cyc_input_img_temp,cyc_input_txt_temp = sess.run([self.fake_cate_fea_img,self.fake_cate_fea_txt,self.cyc_input_img,self.cyc_input_txt],feed_dict= \
                {self.input_GAN_img:self.input_img_data,self.input_GAN_txt:self.input_txt_data,self.keep_prob: 1})
            fake_cate_img_list_data = np.append(fake_cate_img_list_data,fake_cate_img_temp[-remainNum:],axis = 0)
            fake_cate_txt_list_data = np.append(fake_cate_txt_list_data,fake_cate_txt_temp[-remainNum:],axis = 0)
            cyc_input_img_list_data = np.append(cyc_input_img_list_data,cyc_input_img_temp[-remainNum:],axis = 0)
            cyc_input_txt_list_data = np.append(cyc_input_txt_list_data,cyc_input_txt_temp[-remainNum:],axis = 0)

        input_util.save_fea_to_txt(queryImageOutfile,fake_cate_img_list_query)
        input_util.save_fea_to_txt(queryTextOutfile,fake_cate_txt_list_query)
        input_util.save_fea_to_txt(dataImageOutfile,fake_cate_img_list_data)
        input_util.save_fea_to_txt(dataTextOutfile,fake_cate_txt_list_data)
                  
        ret = eng.map(queryImageOutfile,queryTextOutfile,dataImageOutfile,dataTextOutfile,data_dir+"img_train_list.txt",data_dir+"img_test_list.txt",unseenPer,nargout=4)
        print('Unseen MAP:\tI->T:'+str(ret[2])+"\t"+"T->I:"+str(ret[3])) 
        print('Seen MAP:\tI->T:'+str(ret[0])+"\t"+"T->I:"+str(ret[1])) 


def main(_):

    print('GPU:', FLAGS.GPU)
    print('dataset:',dataset )
    print('zeroShot:', zeroShot)
    print('train_dropout:', train_dropout)
    print('batch_size:', batch_size)
    print('learn_rate:', learn_rate)
    print('decay_rate:', decay_rate)
    print('decay_steps:', decay_steps)
    print('total_epoch:', total_epoch)
    model = DADN()
    model.train()

if __name__ == '__main__':
    tf.app.run() 
