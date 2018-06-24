import tensorflow as tf
import numpy as np
import time
from scipy.misc import imread,imsave
from utils import corrupt_image
import os
import random

# m_layer: middle layer for conv+bn+relu
def dn_cnn(input,output_channels,m_layer,is_training=True):
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(input,filters=64,kernel_size=3,padding='same',activation=tf.nn.relu)
    for num in range(2,m_layer+2):
        with tf.variable_scope('block%d' % num):
            #conv2d
            output = tf.layers.conv2d(output,64,3,padding='same', name='conv%d' % num, use_bias=False)
            # bn and relu
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    with tf.variable_scope('block%d' % (m_layer+2)):
        output = tf.layers.conv2d(output, output_channels, 3, padding='same')
    return input-output


class mydenoiser(object):
    def __init__(self,session,input_channel,layer,batch_size,percent):
        self.session=session
        self.input_channel=input_channel
        self.layer=layer
        self.batch_size=batch_size
        self.percent=percent
        # model building
        self.is_training = tf.placeholder(tf.bool,name='is_training')
        # raw clean image
        self.X = tf.placeholder(tf.float32,[None,None,None,self.input_channel],name='raw_image')
        # corrupted by project2's constraint
        # self.Y= tf.placeholder(tf.float32,[None,None,None,self.input_channel])
        self.Y= tf.placeholder(tf.float32,[None,None,None,self.input_channel])
        self.Yprime=dn_cnn(self.Y,self.input_channel,self.layer,is_training=self.is_training)

        
        # self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Yprime - self.Y)
        self.loss=tf.Variable(0.0,tf.float32)
        # for batch in range(batch_size):
        for batch in range(batch_size):
            for channel in range(input_channel):
                self.loss=self.loss+tf.norm(tf.reshape(self.Yprime[batch,:,:,channel],[-1])-tf.reshape(self.Y[batch,:,:,channel],[-1])+tf.reshape(self.X[batch,:,:,channel],[-1]),2)
        self.loss=self.loss/self.batch_size
        # self.loss=(1.0 / batch_size)*tf.nn.l2_loss(self.Yprime - (self.Y-self.X))
        # self.loss=(1.0 / batch_size) *self.loss
        # self.loss_arr=[(1.0 / batch_size)*tf.nn.l2_loss(self.Yprime[k] - (self.Y[k]-self.X[k])) for k in range(batch_size)]
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.session.run(init)

        print("[*] Initialize model success!")

    def train(self,data,lr,epoch):
        batch_num=int(data.shape[0]/self.batch_size)
        iter_num=0
        start_epoch=0
        start_step=0

        print("[*] Start training, epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        print("[*] Start time : ",time.asctime( time.localtime(start_time)))
        for i_epoch in range(start_epoch,epoch):
            np.random.shuffle(data)
            for batch_id in range(start_step,batch_num):
                batch_images=data[batch_id*self.batch_size:(batch_id+1)*self.batch_size,:,:,:]
                corrupt_images=corrupt_image(batch_images,self.percent)
                _,loss,tmp_img= self.session.run([self.train_op, self.loss,self.Y[batch_id]-self.Yprime[batch_id]],
                                                 feed_dict={self.X:batch_images,self.Y:corrupt_images,self.lr:lr[i_epoch],self.is_training:True})
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (i_epoch+1,batch_id+1,batch_num,time.time()-start_time,loss))
                tmp_img = tmp_img.reshape((180,180))
                formatted = (tmp_img * 255 / np.max(tmp_img)).astype('uint8')
                imsave("../result/"+"epoch:"+str(i_epoch)+"random:"+str(batch_id)+"_result.png",formatted)
            
            # denoised=self.Y-self.Yprime
            
                
        ckpt_dir='../backup/'
        self.save(epoch, ckpt_dir,model_name='DnCNN-tensorflow'+str(epoch))
        print("[*] Save Done. Finish training.")

    def denoise(self,data):
        output_clean_image=self.session.run(self.Y-self.Yprime,feed_dict={self.Y:data,self.is_training:False})
        return output_clean_image
    
    def save(self, iter_num, ckpt_dir, model_name='DnCNN-tensorflow'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.session,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.session, full_path)
            return True, global_step
        else:
            return False, 0

    def test(self, test_file_name, ckpt_dir):
        # init variables
        tf.initialize_all_variables().run()
        assert len(test_file_name) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        img=np.array(imread("../data/"+test_file_name+".png",0))
        img=img.reshape([1,img.shape[0],img.shape[1],self.input_channel])
        denoised=self.denoise(img)
        if self.input_channel==1:
            denoised=denoised.reshape([denoised.shape[1],denoised.shape[2]])
        elif self.input_channel==3:
            denoised=denoised.reshape([denoised.shape[1],denoised.shape[2],3])
        imsave("../result/"+test_file_name+"_result2.png",denoised)
        print(" [*] Save SUCCESS...")