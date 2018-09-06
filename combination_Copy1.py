# coding: utf-8
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

from tqdm import tqdm
import os
import os.path
import multiprocessing
import time
import sys

import numpy as np
import tensorflow as tf
import gensim
import math

sys.path.append(".\\vgg")   #路径列表中添加vgg模块的路径
import input_data
import  my_vgg
import tools
import pickle


# def unpickle(file):
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


pic = ".\\vgg\\cifar-10-batches-py\\batches.meta"
data_dir = ".\\vgg\\cifar-10-batches-py\\data_batch_"

meta = unpickle(pic)  #data format  -->  dictionary

train_path = [ (data_dir + str(i)) for i in np.arange(1, 5)]
val_path = ".\\vgg\\cifar-10-batches-py\\data_batch_5"
test_path =  ".\\vgg\\cifar-10-batches-py\\test_batch"

print(test_path)
x =tools.unpickle(test_path)
print(x.keys())



###parameters
IMG_W = 32
IMG_H = 32
N_CLASSES = 10
BATCH_SIZE = 1
learning_rate = 0.01
MAX_STEP = 10000   # it took me about one hour to complete the training.
IS_PRETRAIN = True
word_vector_dim = 200

# model = gensim.models.Word2Vec.load("./gensim/w2v_database/wiki.en.text.model")   ###read w2v model from path
model = gensim.models.Word2Vec.load('.\\gensim\\text8.model')
print((model["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]).shape)


###load vgg visual model graph
label_string = unpickle('.\\vgg\\cifar-10-batches-py\\batches.meta')['label_names']
# cifar-10的标签词汇对应的词向量,是一个10*200的矩阵
label_string_vector = model[label_string]
print(len(model.wv.vocab))
print(label_string)

# 词汇表中所有单词对应的词向量
tmparray = model[ (model.wv.vocab).keys() ]
keys =list((model.wv.vocab).keys() )    #词汇表中的71290个单词
 
###print labels and their ids in the w2v model
# 为cifar10的10个标签初始化一个数组，存储它们在词汇表中对应的标签
id_array = np.zeros(N_CLASSES)

word_id = 0
pointer = 0
#j遍历 是cifar_10标签词汇中对应的单词
for j in label_string:
    word_id = 0
    #i是遍历词汇表中71290个词对应的单词
    for i in (model.wv.vocab).keys():    
        if i==j:
            print('word = ',i,',       id = ',word_id)
            #在词汇表中找到标签名称，就将标签名称在词汇表中对应的id赋值给标签值在id_array对应的位置的值
            id_array[pointer] = word_id
            pointer = pointer + 1
            break
        word_id = word_id + 1
print("--end extract id--")


##vector to word
#reference::https://stackoverflow.com/questions/32759712/how-to-find-the-closest-word-to-a-vector-using-word2vec
topn = 1
for i in label_string_vector:
    #print(i.shape)
    #和cifar10标签值对应的词向量最相似的一个单词
    most_similar_words = model.most_similar( [i] , [], topn)
    print(most_similar_words)

# 训练
def train():
    with tf.Graph().as_default():
        log_dir = '.\\vgg\\my_train_logs\\'

        train_log_dir = '.\\devise_logs\\train\\'
        val_log_dir = '.\\devise_logs\\val\\'


        #images
        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3],name = 'input_x')

        #labels
        y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES],name = 'input_y_')
        

        #     logits:vgg16中fc8层的输出
        '''
         x = tools.FC_layer('fc8', x, out_nodes=n_classes)
    
                   return x
         '''
        logits = my_vgg.VGG16N(x, N_CLASSES, is_pretrain =False)
        saver = tf.train.Saver(tf.global_variables())
        
        id_array_tensor = tf.placeholder(tf.int32 ,[N_CLASSES])
        
        
        
        #原定义的VGG16的最后一层输出结果和标签值对比
        softmax_accuracy = tools.accuracy(logits, y_)
        vgg_predict = tf.argmax(logits,-1)
        #从vgg_predict中找到id_array_tensor 对应的索引，并将索引位置的值返回
        vgg_predict_id = tf.gather_nd(id_array_tensor,vgg_predict)
        

        #声明标签值对应的词向量tensor
        get_label_string_tensor = tf.placeholder(tf.float32 , shape = [None,word_vector_dim],name = "label_string")##word_vector_dim
        #标签值对应的词向量的Tensor和标签值进行矩阵的乘积
        tmp = tf.matmul(tf.cast(y_,tf.float32),get_label_string_tensor)
        #print(tmp.shape)

        initializer = tf.contrib.layers.variance_scaling_initializer()
        #取出VGG16的4096层，去掉后面的softmax层
        # fc7 = tf.get_default_graph().get_tensor_by_name("VGG16/fc7/Relu:0")
        fc6 = tf.get_default_graph().get_tensor_by_name("VGG16/fc6/Relu:0")
        #添加映射层，将1*1*4096转化为1*1*1024
        fc8 = tf.layers.dense(inputs = fc6, units = 2000, kernel_initializer = initializer, name = "combination_hidden1")
        #再添加一层，将1*1*1024转换为1*1*200（word_vector_dim)
        image_feature_output = tf.layers.dense(inputs = fc8, units = word_vector_dim, kernel_initializer = initializer, name = "combination_hidden2")


        #devise_loss
        #声明joint_model的loss计算方式

        tmparray_tensor = tf.placeholder(tf.float32 , shape = [None,word_vector_dim],name = 'tmparray')

        margin = 0.1
        #tMV here means that max (tJ *M* V - tLabel *M *V ,0 ) in essay
        #tmparray_tensor mearns tJ     and tmp means tmp

        tMV = tf.nn.relu( margin + tf.matmul((tmparray_tensor - tmp),tf.transpose(tf.cast(image_feature_output,tf.float32))))
        hinge_loss = tf.reduce_mean(tf.reduce_sum(tMV,0) , name = 'hinge_loss')

        train_step1 =tf.train.AdamOptimizer(0.0001, name="optimizer").minimize(hinge_loss)

        #tMV_ here means that tJ *M* V in essay
        tMV_ = tf.matmul(tmparray_tensor,tf.transpose(tf.cast(image_feature_output,tf.float32)))

        #accuracy
        predict_label = tf.argmax(tMV_, 0)
        predict_label = tf.cast(predict_label, tf.int32)
        predict_label = tf.reshape(predict_label,[-1,1],name = 'predict_label_text')

        #id_array_tensor = tf.placeholder(tf.int32 ,[N_CLASSES])
        select_id = tf.cast(tf.argmax(input = y_, axis = -1),tf.int32)
        select_id = tf.reshape(select_id,[1])
        y_label = tf.gather_nd(id_array_tensor,select_id)
        y_label = tf.reshape(y_label,[-1,1],name ='true_label_text')

        print(y_label.shape)
        print(predict_label.shape)


        acc,acc_op = tf.metrics.accuracy(labels = y_label, predictions = predict_label
                                         , weights=None, metrics_collections=None
                                         , updates_collections=None, name="acc")
        
        summary_op = tf.summary.merge_all()
        saver2 = tf.train.Saver(tf.global_variables(),max_to_keep=15)
        
        with tf.Session() as sess:
            tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
            val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)
            # 合并所有概要算子
            merged = tf.summary.merge_all()
        
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            
            print("Reading vgg16 checkpoints...")
            module_file = tf.train.latest_checkpoint('.\\vgg\\my_train_logs\\')
            saver.restore(sess, module_file)

# 成功加载预训练的vgg模型，开始训练joint model
##---------------------------------------------------------------Training-------------------------------------------------------------------------------------------------
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            print('\Triaining....')
            module_file2 = tf.train.latest_checkpoint('.\\devise_logs\\train\\')
            saver2.restore(sess, module_file2)
            try:
                vgg_total_correct = 0
                for step in tqdm(range(MAX_STEP)):
                    if coord.should_stop():
                            break

                    for batch_path in train_path:
                        print("batch_path:%s" % str(batch_path))
                        batch=batch_path.split('_')[-1]

                        tra_images = tools.unpickle(batch_path)[b'data']
                        tra_images = np.reshape(tra_images, (-1, 3, 32, 32))
                        tra_images = np.transpose(tra_images, [0, 2, 3, 1])

                        tra_labels =tools.unpickle(batch_path)[b'labels']
                        tra_labels_tmp = np.max(tra_labels) + 1
                        tra_labels = np.eye(tra_labels_tmp)[tra_labels]  # 将标签转换为one-hot类型

                        i = 0
                        j = BATCH_SIZE
                        for k in range(int(math.ceil(len(tra_images) / BATCH_SIZE))):
                            batch_imgs, batch_labels = tra_images[i:j], tra_labels[i:j]
                            feed_dict = {get_label_string_tensor: label_string_vector, tmparray_tensor: tmparray,
                                         id_array_tensor: id_array, x: batch_imgs, y_: batch_labels}
                            summary = sess.run(merged, feed_dict=feed_dict)

                            loss,_,accuracy_,acc_operator,predict_label_,y_label_,summary_str,vgg_correct,vgg_predict_id_ = sess.run(
                                                      [hinge_loss,train_step1,acc,acc_op,predict_label,y_label,summary_op,softmax_accuracy,vgg_predict_id],
                                                      feed_dict =feed_dict)
                            tra_summary_writer.add_summary(summary,step)

                            i+=BATCH_SIZE
                            j+=BATCH_SIZE

                            #print(vgg_correct)
                            if vgg_correct > 50 :
                                vgg_total_correct = vgg_total_correct + 1

                            if k%20 == 0:
                                print("step %d"%step)
                                print('%d / %d steps'%(step,MAX_STEP),'loss = ',loss,'    acc = ',accuracy_,'\n\n')
                                print ('vgg predict acc  ',vgg_total_correct*1.0/(step+1),' ---->vgg predict     ',keys[int(vgg_predict_id_)])
                                print ('           devise predict_label',predict_label_,' ---->DeVise predict  ',keys[int(predict_label_)])
                                print ('                y_label             ',y_label_,      ' ---->ground  truth is',keys[int(y_label_)],'\n\n-------\n\n')
                            if k%200==0:
                                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                                saver2.save(sess, checkpoint_path, global_step=k)

                        checkpoint_path = os.path.join(train_log_dir, 'model_batch.ckpt')
                        saver2.save(sess, checkpoint_path, global_step=batch)


                    if (step+1) % 5 == 0 or (step + 1) == MAX_STEP:
                        checkpoint_path = os.path.join(train_log_dir, 'model_epoch.ckpt')
                        saver2.save(sess, checkpoint_path, global_step=step)
                
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
                coord.join(threads)

        print("end training\n\n\n------------------------------------------\n")



def test():
    log_dir = '.\\vgg\\my_train_logs\\'
    with tf.Graph().as_default():
        # images
        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3], name='input_x')

        # labels
        y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES], name='input_y_')

        #     logits:vgg16中fc8层的输出
        '''
         x = tools.FC_layer('fc8', x, out_nodes=n_classes)
    
                   return x
         '''
        logits = my_vgg.VGG16N(x, N_CLASSES, is_pretrain=False)
        saver = tf.train.Saver(tf.global_variables())

        id_array_tensor = tf.placeholder(tf.int32, [N_CLASSES])

        # 原定义的VGG16的最后一层输出结果和标签值对比
        softmax_accuracy = tools.accuracy(logits, y_)
        vgg_predict = tf.argmax(logits, -1)
        # 从vgg_predict中找到id_array_tensor 对应的索引，并将索引位置的值返回
        vgg_predict_id = tf.gather_nd(id_array_tensor, vgg_predict)

        # 声明标签值对应的词向量tensor
        get_label_string_tensor = tf.placeholder(tf.float32, shape=[None, word_vector_dim],
                                                 name="label_string")  ##word_vector_dim
        # 标签值对应的词向量的Tensor和标签值进行矩阵的乘积
        tmp = tf.matmul(tf.cast(y_, tf.float32), get_label_string_tensor)
        # print(tmp.shape)

        initializer = tf.contrib.layers.variance_scaling_initializer()
        # 取出VGG16的4096层，去掉后面的softmax层
        # fc7 = tf.get_default_graph().get_tensor_by_name("VGG16/fc7/Relu:0")
        fc6 = tf.get_default_graph().get_tensor_by_name("VGG16/fc6/Relu:0")
        # 添加映射层，将1*1*4096转化为1*1*1024
        fc8 = tf.layers.dense(inputs=fc6, units=2000, kernel_initializer=initializer, name="combination_hidden1")
        # 再添加一层，将1*1*1024转换为1*1*200（word_vector_dim)
        image_feature_output = tf.layers.dense(inputs=fc8, units=word_vector_dim, kernel_initializer=initializer,
                                               name="combination_hidden2")

        # devise_loss
        # 声明joint_model的loss计算方式

        tmparray_tensor = tf.placeholder(tf.float32, shape=[None, word_vector_dim], name='tmparray')

        margin = 0.1
        # tMV here means that max (tJ *M* V - tLabel *M *V ,0 ) in essay
        # tmparray_tensor mearns tJ     and tmp means tmp

        tMV = tf.nn.relu(
            margin + tf.matmul((tmparray_tensor - tmp), tf.transpose(tf.cast(image_feature_output, tf.float32))))
        hinge_loss = tf.reduce_mean(tf.reduce_sum(tMV, 0), name='hinge_loss')

        train_step1 = tf.train.AdamOptimizer(0.0001, name="optimizer").minimize(hinge_loss)

        # tMV_ here means that tJ *M* V in essay
        tMV_ = tf.matmul(tmparray_tensor, tf.transpose(tf.cast(image_feature_output, tf.float32)))

        # accuracy
        predict_label = tf.argmax(tMV_, 0)
        predict_label = tf.cast(predict_label, tf.int32)
        predict_label = tf.reshape(predict_label, [-1, 1], name='predict_label_text')

        # id_array_tensor = tf.placeholder(tf.int32 ,[N_CLASSES])
        select_id = tf.cast(tf.argmax(input=y_, axis=-1), tf.int32)
        select_id = tf.reshape(select_id, [1])
        y_label = tf.gather_nd(id_array_tensor, select_id)
        y_label = tf.reshape(y_label, [-1, 1], name='true_label_text')

        print(y_label.shape)
        print(predict_label.shape)

        acc, acc_op = tf.metrics.accuracy(labels=y_label, predictions=predict_label
                                          , weights=None, metrics_collections=None
                                          , updates_collections=None, name="acc")

        summary_op = tf.summary.merge_all()
        # 训练结束，开始测试模型
##-----------------------------------------------------------------Testing---------------------------------------
        with tf.Session() as sess:
            print("Reading vgg16 checkpoints...")
            module_file = tf.train.latest_checkpoint('.\\vgg\\my_train_logs\\')
            saver.restore(sess, module_file)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            print('----Testing----')
            saver2=tf.train.Saver(tf.global_variables())
            print("Reading devise checkpoints...")
            # 参数恢复
            module_file = tf.train.latest_checkpoint('.\\devise_logs\\train\\')
            saver2.restore(sess, module_file)

            test_images =tools.unpickle(test_path)[b'data']
            test_images = np.reshape(test_images, (-1, 3, 32, 32))
            test_images = np.transpose(test_images, [0, 2, 3, 1])

            test_labels =tools.unpickle(test_path)[b'labels']
            test_labels_tmp = np.max(test_labels) + 1
            test_labels = np.eye(test_labels_tmp)[test_labels]  # 将标签转换为one-hot

            sess.run(tf.local_variables_initializer())
            #加载训练好的模型后开始测试

            i = 0
            j = 1
            correct_num=0

            for k in range(int(math.ceil(len(test_images) / 1))):
                batch_imgs, batch_labels = test_images[i:j], test_labels[i:j]

                loss,accuracy_,acc_operator,predict_label_,y_label_ = sess.run([hinge_loss,acc,acc_op,predict_label,y_label],
                                          feed_dict = {get_label_string_tensor:label_string_vector,tmparray_tensor:tmparray,
                                                       id_array_tensor:id_array,x:batch_imgs, y_:batch_labels})
                get_acc = accuracy_
                i += 1
                j += 1

                print('k = ',k, 'loss = ', loss, '    acc = ', accuracy_, '\n')
                # print('vgg predict acc  ', vgg_total_correct * 1.0 / (step + 1), ' ---->vgg predict     ',
                #       keys[int(vgg_predict_id_)])
                print('devise predict_label', predict_label_, ' ---->DeVise predict  ',
                      keys[int(predict_label_)])
                print('y_label             ', y_label_, ' ---->ground truth is',
                      keys[int(y_label_)], '\n-------\n')
                if int(predict_label_)==int(y_label_):
                    correct_num+=1

            print('准确率为： ',correct_num/len(test_images))
                



t0 = time.time()
print(t0)
# train()
test()
t1 = time.time()




