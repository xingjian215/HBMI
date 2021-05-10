import os
import gensim
import torch.nn as nn
import os
from data import train_data_prepare,load_data
from test import test_and_valid_data_prepare,test
from train import train
import torch
import shutil
import itertools
import random
import numpy as np
import pandas as pd
import re
import json

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def restore_net(saved_model_file):
    model = torch.load(saved_model_file)     #load xxx.pkl file and assign its contents to model
    return model

para_dict = {'batch_size':8,
        'lr':0.0001,
        'weight_decay':1e-4,
        'EPOCH':20,
        'transformer_num_layers':3,
        'num_heads':2,
        'dropout':0.5}

def run():
    train_dataset_file = 'C:/Users/lenovo/PycharmProjects/me/train.tsv'
    valid_dataset_file = 'C:/Users/lenovo/PycharmProjects/me/valid.tsv'
    test_dataset_file = 'C:/Users/lenovo/PycharmProjects/me/test.tsv'
    word2vec_preweight_file = 'C:/Users/lenovo/PycharmProjects/me/word2vec_preweight_file.txt'
    train_statement_file = 'C:/Users/lenovo/PycharmProjects/ex-bert/output_dir/train_statement.json'
    train_meta_file = 'C:/Users/lenovo/PycharmProjects/ex-bert/output_dir/train_meta.json'
    valid_statement_file = 'C:/Users/lenovo/PycharmProjects/ex-bert/output_dir/valid_statement.json'
    valid_meta_file = 'C:/Users/lenovo/PycharmProjects/ex-bert/output_dir/valid_meta.json'
    test_statement_file = 'C:/Users/lenovo/PycharmProjects/ex-bert/output_dir/test_statement.json'
    test_meta_file = 'C:/Users/lenovo/PycharmProjects/ex-bert/output_dir/test_meta.json'


    word2num, word2vec_preweight, train_samples = train_data_prepare(train_dataset_file, word2vec_preweight_file,train_statement_file,train_meta_file)
    valid_samples = test_and_valid_data_prepare(valid_dataset_file, word2num,valid_statement_file,valid_meta_file)
    test_samples = test_and_valid_data_prepare(test_dataset_file, word2num,test_statement_file,test_meta_file)


    list_1 = [0,1,2,3,4,5,6]
    array_list = itertools.permutations(list_1)
    res_list = []
    for one in array_list:
        res_list.append(list(one))
    res_list=[[0]]
    result_file_transformer =  './result_file_transformer_pad_cls.txt'
    # random.seed(1)
    # random_seed = random.sample(range(1, 10000), N)     #初始化随机种子
    num_layers = [3]
    num_heads = [2]
    result = []
    temp_max=0
    for i1 in range(len(num_layers)):
        for j1 in range(len(num_layers)):
            para_dict['transformer_num_layers'] = num_layers[i1]
            para_dict['num_heads'] = num_heads[j1]

            out_dir = './Encoder_4'
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.mkdir(out_dir)
            result_df_file = out_dir + '/Encoder_2.xlsx'

            with open(result_file_transformer, 'a+') as result_f:
                for one in range(0,len(res_list)):
                    #out_label_save_file = out_dir+'out_label_save_file'
                    train_process_save_file = out_dir+'/train_process_save_file'+str(one)
                    jpg_file = out_dir+'/jpg_file'+str(one)
                    result_test_label_file = out_dir+'/test_label.txt'
                    save_model_file = out_dir+'/model_file'+str(one)
                    test_acc = []
                    test_acc_max = []
                    N = 10
                    #N = int(input('Input the number of model:'))
                    for i in range(N):
                        #setup_seed(i)
                        train_process_save_file_1 = train_process_save_file+str(i)+'.txt'
                        jpg_file_1 = jpg_file+str(i)+'.jpg'
                        save_model_file_1 = save_model_file+str(i)+'.pkl'
                        process_file_1 = open(train_process_save_file_1, 'w')

                        test_acc1,test_acc_max1,temp_max_return = train(train_samples, valid_samples, test_samples,res_list[one],word2vec_preweight, len(word2num), process_file_1, jpg_file_1,save_model_file_1,result_test_label_file,para_dict,temp_max)
                        test_acc.append(test_acc1)
                        test_acc_max.append(test_acc_max1)
                        if temp_max < temp_max_return:
                            temp_max=temp_max_return
                            print('temp_max is: %f' % temp_max)
                        if temp_max<test_acc_max1:
                            temp_max = test_acc_max1

                        process_file_1.write('model '+str(i)+ ' test accuracy:' + str(test_acc1))
                        print('model  '+str(i)+ ' test accuracy:' + str(test_acc1)+'-'+ str(test_acc_max1))
                        process_file_1.close()
                    # caculate average
                    avg_acc = sum(test_acc)/len(test_acc)
                    avg_acc_max = sum(test_acc_max) / len(test_acc_max)
                    test_acc.append(avg_acc)
                    result.append(test_acc)
                    print('=====>The average test accuracy of  model for '+str(res_list[one])+' is:'+str(avg_acc)+'-'+str(avg_acc_max))
                    result_f.writelines(json.dumps(para_dict) + ':'+str(res_list[one])+str(avg_acc)+'\r\n')
                result_df = pd.DataFrame(result)
                result_df.index = [re.sub('\D', '', str(j)) for j in res_list]
                column = [str(i) for i in range(N)]
                column.append('avg_Acc')
                result_df.columns = column
                result_df.to_excel(result_df_file)

if __name__ == '__main__':
    run()
