'''
Purpose of making this class is to load EMA data into TacoTron 2
'''

import os
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import HTK
from copy import deepcopy
from scipy.io import loadmat
import pickle

def read_data():
    EmaDir = 'EmaClean/'
    AliDir = 'ForceAlign/'
    emafiles = sorted(os.listdir(EmaDir))
    alifiles = sorted(os.listdir(AliDir))
    train_ema = [loadmat(EmaDir+idx) for idx in emafiles]
    train_ali = [pd.read_csv(AliDir+idx,header=None) for idx in alifiles]

    return (train_ali, train_ema)
count = 0


def pre_process(index, train = True):
        # train_ema = self.train_ema
        # train_ali = self.train_ali
        # print("This is length of phoneme in pre_process", len(train_ali))
        # phoneme=[]
        # new_phoneme=[]
        with open('variables/new_phoneme', 'rb') as handle:
            new_phoneme = pickle.loads(handle.read())
        
        with open('variables/word_to_int', 'rb') as handle:
            word_to_int = pickle.loads(handle.read())
    
        with open('variables/set_phoneme', 'rb') as handle:
            set_phoneme = pickle.loads(handle.read())
    
        with open('variables/time_phoneme', 'rb') as handle:
            time_phoneme = pickle.loads(handle.read())

        with open('variables/time_sil', 'rb') as handle:
            time_sil = pickle.loads(handle.read())

        with open('variables/int_to_word', 'rb') as handle:
           int_to_word = pickle.loads(handle.read())

        with open('variables/max_len_ema', 'rb') as handle:
            max_len_ema = pickle.loads(handle.read())
        
        with open('variables/new_ema', 'rb') as handle:
           new_ema = pickle.loads(handle.read())

        with open('variables/maxlen_phoneme', 'rb') as handle:
            maxlen_phoneme = pickle.loads(handle.read())

        with open('variables/train_new_ema', 'rb') as handle:
            train_new_ema = pickle.loads(handle.read())

        with open('variables/test_new_ema', 'rb') as handle:
            test_new_ema = pickle.loads(handle.read())

        with open('variables/train_new_phoneme', 'rb') as handle:
            train_new_phoneme = pickle.loads(handle.read())

        with open('variables/test_new_phoneme', 'rb') as handle:
            test_new_phoneme = pickle.loads(handle.read())

        with open('variables/train_ema_len', 'rb') as handle:
            train_ema_len = pickle.loads(handle.read())

        with open('variables/test_ema_len', 'rb') as handle:
            test_ema_len = pickle.loads(handle.read())

        with open('variables/train_phoneme_len', 'rb') as handle:
            train_phoneme_len = pickle.loads(handle.read())

        with open('variables/test_phoneme_len', 'rb') as handle:
            test_phoneme_len = pickle.loads(handle.read())




        if train:
            return [train_new_phoneme[index], train_new_ema[index]]
        else:
            return [test_new_phoneme[index], test_new_ema[index]]




class train_ema(torch.utils.data.Dataset):
    def __init__(self):
        # self.train_ema, self.train_ali = read_data()
        self.phoneme, self.ema = read_data()
        # print("Data imported, from read_data into audiopath, sample length can be", len(self.phoneme[0]))
        # print("And that phoneme is", self.phoneme[0])

        
    

    def get_mel_text_pair(self, index):
        # separate filename and text
        
        phoneme, ema = pre_process(index, train=True)
        return (phoneme, ema)

    def __getitem__(self, index):
        global count
        
        # print("is this even getting executed", count)
        count+=1
        return self.get_mel_text_pair(index)

    def __len__(self):
        return 400



class test_ema(torch.utils.data.Dataset):

    def __init__(self):
        # self.train_ema, self.train_ali = read_data()
        self.phoneme, self.ema = read_data()
        # print("Data imported, from read_data into audiopath, sample length can be", len(self.phoneme[0]))
        # print("And that phoneme is", self.phoneme[0])

        
    

    def get_mel_text_pair(self, index):
        # separate filename and text
        
        phoneme, ema = pre_process(index, train=False)
        return (phoneme, ema)

    def __getitem__(self, index):
        global count
        
        # print("is this even getting executed", count)
        count+=1
        return self.get_mel_text_pair(index)
    

    def __len__(self):
        return 60
    







