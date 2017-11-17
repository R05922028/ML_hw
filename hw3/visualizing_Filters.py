#!/usr/bin/env python
# -- coding: utf-8 --
import os
import sys
import argparse
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
#from termcolor import colored,cprint
import numpy as np
#from utils import *
import pickle as pkl

'''basedir = os.path.dirname(os.path.dirname(os.path.realpath(_file)))
exp_dir = 'exp'
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
vis_dir = os.path.join('image','vis_layer')
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)
filter_dir = os.path.join('image','vis_filter')
if not os.path.exists(filter_dir):
    os.makedirs(filter_dir)'''



nb_class = 7
LR_RATE = 1e-2
NUM_STEPS = 200
RECORD_FREQ = 10

def deprocess_image(x):
    """
    As same as that in problem 4.
    """
    return x

'''parser = argparse.ArgumentParser(prog='visFilter.py',
        description='Visualize CNN filter.')
parser.add_argument('--model',type=str,default='simple',choices=['simple','NTUEE'],
        metavar='<model>')
parser.add_argument('--epoch',type=int,metavar='<#epoch>',default=20)
parser.add_argument('--mode',type=int,metavar='<visMode>',default=1,choices=[1,2])
parser.add_argument('--batch',type=int,metavar='<batch_size>',default=64)
parser.add_argument('--idx',type=int,metavar='<suffix>',required=True)'''
args = 1
'''store_path = "{}_epoch{}{}".format(args.model,args.epoch,args.idx)
print(colored("Loading model from {}".format(storepath),'yellow',attrs=['bold']))
modelpath = os.path.join(expdir,storepath,'model.h5')'''
emotion_classifier = load_model('final_model.h5')


layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])

for layer in emotion_classifier.layers[1:]:
    print(layer.name)

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def grad_ascent(num_step,input_image_data,iter_func):
    """
    Implement this function!
    """
    filter_images = iter_func([input_image_data])

    for i in range(num_step):
        filter_images = iter_func([filter_images[1]])
        print(filter_images)

    return filter_images

input_img = emotion_classifier.input
# visualize the area CNN see
if True:
    collect_layers = list()
    collect_layers.append(K.function([input_img,K.learning_phase()],[layer_dict['conv2d_3'].output]))

    f = [open('X_train.pkl', 'rb'), open('Y_train.pkl', 'rb')]
    X_train = pkl.load(f[0])
    Y_train = pkl.load(f[1])
    X_train = [np.fromstring(X_train[i], dtype=int, sep=' ') for i in range(len(X_train))]
    X_train = np.array(X_train).reshape(-1, 48*48)
    #dev_feat = load_pickle('fer2013/test_with_ans_pixels.pkl')
    #dev_label = load_pickle('fer2013/test_with_ans_labels.pkl')
    choose_id = 777
    photo = X_train[choose_id]
    '''photo = photo.split()
    for p in photo:
        p = int(p)
    photo = np.array(photo)'''
    for cnt, fn in enumerate(collect_layers):
        im = fn([photo.reshape(1,48,48,1),0])
        fig = plt.figure(figsize=(14,8))
        nb_filter = im[0].shape[3]
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/16,16,i+1)
            ax.imshow(im[0][0,:,:,i],cmap='PuBu')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
            #plt.show()
        fig.suptitle('Output of layer{} (Given image{})'.format(cnt,choose_id))
        #img_path = os.path.join(vis_dir,store_path)
        img_path = './'
        #if not os.path.isdir(img_path):
        #    os.mkdir(img_path)
        fig.savefig(os.path.join(img_path,'layer{}'.format(cnt)))

else:
    name_ls = ['conv2d_3']
    collect_layers = list()
    collect_layers.append(layer_dict[name_ls[0]].output)

    for cnt, c in enumerate(collect_layers):
        filter_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]
        nb_filter = c.shape[-1]
        for i in range(len(filter_imgs)):
            for j in range(nb_filter):
                filter_imgs[i].append([])
        for filter_idx in range(nb_filter):
            input_img_data = np.random.random((1, 48, 48, 1))
            loss = K.mean(c[:, :, :, filter_idx])
            grads = normalize(K.gradients(loss,input_img)[0])
            iterate = K.function([input_img],[loss,grads])


            #"You need to implement it."
            for num_step in range(NUM_STEPS//RECORD_FREQ):
                filter_imgs[num_step][filter_idx] = (grad_ascent((num_step+1) * RECORD_FREQ, input_img_data, iterate))


        for it in range(NUM_STEPS//RECORD_FREQ):
            fig = plt.figure(figsize=(14,8))
            for i in range(nb_filter):
                ax = fig.add_subplot(int(nb_filter)/16,16,i+1)
                ax.imshow(filter_imgs[it][i][1].reshape(48,48),cmap='Green')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.xlabel('{:.3f}'.format(filter_imgs[it][i][0]))
                plt.tight_layout()
                #plt.show()
            fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[0],it*RECORD_FREQ))
            #img_path = os.path.join(filter_dir,'{}-{}'.format(store_path,name_ls[0]))
            img_path = './'
            if not os.path.isdir(img_path):
                os.mkdir(img_path)
            fig.savefig(os.path.join(img_path,'e{}'.format(it*RECORD_FREQ)))

