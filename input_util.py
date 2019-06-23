# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import gc
import random


def read_aux_text_feature_from_txt(filepath):
    txt_file = open(filepath,'r')
    txt_fea = []
    while True:  
        line = txt_file.readline() 
        if line : 
            txt_fea.append(line.split())
        else:
            break
    txt_file.close()
    return txt_fea


def read_feature_from_txt_pair(input_img_fea_path,label_img_path,input_txt_fea_path,label_txt_path,cate_fea_path,is_train,dataset,unseenPer):
    input_img_fea_txt_file = open(input_img_fea_path,'r')
    lab_img_file = open(label_img_path,'r')
    input_txt_fea_txt_file = open(input_txt_fea_path,'r')
    lab_txt_file = open(label_txt_path,'r')

    input_img_fea = []
    input_txt_fea = []
    label = []
    catefeaAll = []
    misCatefeaAll = []
    cnt=0
    total = 0
    print("read_feature_from_txt_pair")

    catefea = read_aux_text_feature_from_txt(cate_fea_path)
    unseenSet = set()
    while True:  
        line1 = input_img_fea_txt_file.readline() 
        line2 = lab_img_file.readline() 
        line3 = input_txt_fea_txt_file.readline() 
        line4 = lab_txt_file.readline() 
        if line1 and line2 and line3 and line4: 

            cate_num_img = int(line2.split()[1])
            cate_num_txt = int(line4.split()[1])

            if cate_num_img != cate_num_txt:
                print("img and txt are not pair")
                return

            if dataset == "xmedianet":
                cate_num_img = cate_num_img-1
                cate_num_txt = cate_num_txt-1

            if is_train:            
                if unseenPer == 0.5:
                    if cate_num_img % 2 == 0: 
                        unseenSet.add(cate_num_img)
                        total = total+1
                        continue
                elif unseenPer == 0.1:
                    if cate_num_img % 10 == 0: 
                        unseenSet.add(cate_num_img)
                        total = total+1
                        continue        
                elif unseenPer == 0.3:
                    if cate_num_img % 3 == 0: 
                        unseenSet.add(cate_num_img)
                        total = total+1
                        continue
                elif unseenPer == 0.7:
                    if cate_num_img % 3 != 0:
                        unseenSet.add(cate_num_img)
                        total = total+1
                        continue
                elif unseenPer == 0.9:
                    if cate_num_img % 10 != 0: 
                        unseenSet.add(cate_num_img)
                        total = total+1
                        continue

            input_img_fea.append(line1.split())
            input_txt_fea.append(line3.split())
            label.append(cate_num_img)
            catefeaAll.append(catefea[cate_num_img])

            cnt=cnt+1
            total = total+1
        else:
            break
    input_img_fea_txt_file.close()
    lab_img_file.close()
    input_txt_fea_txt_file.close()
    lab_txt_file.close()

    print("read data pairs num:"+str(cnt))
    print("Image:",np.array(input_img_fea).shape)
    print("Text:",np.array(input_txt_fea).shape)
    print("Label:",np.array(label).shape)
    print("category feature:",np.array(catefeaAll).shape)
    if is_train:
        print("unseen:",unseenSet)

    return np.array(input_img_fea),np.array(input_txt_fea),np.array(label),np.array(catefeaAll),np.array(misCatefeaAll),cnt,total

def get_batch_shuffle(input_fea,label,total_num,batch_size):
    input_fea_batch = []
    label_batch = []
    for x in random.sample(range(total_num),batch_size): 
        input_fea_batch.append(input_fea[x])
        label_batch.append(label[x])
    input_fea_batch = np.array(input_fea_batch)
    label_batch = np.array(label_batch)
    return input_fea_batch,label_batch

def get_batch_shuffle_pair(input_img_fea,input_txt_fea,label,total_num,batch_size):
    input_img_fea_batch = []
    input_txt_fea_batch = []
    label_batch = []
    for x in random.sample(range(total_num),batch_size): 
        input_img_fea_batch.append(input_img_fea[x])
        input_txt_fea_batch.append(input_txt_fea[x])
        label_batch.append(label[x])
    input_img_fea_batch = np.array(input_img_fea_batch)
    input_txt_fea_batch = np.array(input_txt_fea_batch)
    label_batch = np.array(label_batch)
    return input_img_fea_batch,input_txt_fea_batch,label_batch

def get_batch_shuffle_pair_mis(input_img_fea,input_txt_fea,label,total_num,batch_size):
    input_img_fea_batch = []
    input_txt_fea_batch = []
    input_mis_img_fea_batch = []
    input_mis_txt_fea_batch = []
    label_batch = []
    for x in random.sample(range(total_num),batch_size): 
        input_img_fea_batch.append(input_img_fea[x])
        input_txt_fea_batch.append(input_txt_fea[x])
        label_batch.append(label[x])
        while(True):
            randTmp = random.randint(0, total_num-1)
            if label[randTmp][0] != label[x][0]:
                input_mis_img_fea_batch.append(input_img_fea[randTmp])
                input_mis_txt_fea_batch.append(input_txt_fea[randTmp])
                break

    input_img_fea_batch = np.array(input_img_fea_batch)
    input_txt_fea_batch = np.array(input_txt_fea_batch)
    input_mis_img_fea_batch = np.array(input_mis_img_fea_batch)
    input_mis_txt_fea_batch = np.array(input_mis_txt_fea_batch)
    label_batch = np.array(label_batch)
    return input_img_fea_batch,input_txt_fea_batch,label_batch,input_mis_img_fea_batch,input_mis_txt_fea_batch


def get_batch_seq_pair(input_img_fea,input_txt_fea,label,total_num,cur_num,batch_size):
    input_img_fea_batch = []
    input_txt_fea_batch = []
    label_batch = []
    for x in range(cur_num,min(total_num,cur_num+batch_size)):
        input_img_fea_batch.append(input_img_fea[x])
        input_txt_fea_batch.append(input_txt_fea[x])
        label_batch.append(label[x])
    input_img_fea_batch = np.array(input_img_fea_batch)
    input_txt_fea_batch = np.array(input_txt_fea_batch)
    label_batch = np.array(label_batch)
    return input_img_fea_batch,input_txt_fea_batch,label_batch


def save_fea_to_txt(filepath,r):

    f = open(filepath,"w+")
    r = np.array(r)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            f.write('{:.8f} '.format(r[i][j])) 
        f.write('\n')
    print('successfully saved:'+filepath)
    f.close()    

