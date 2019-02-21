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


def pre_process(index, train = True):
        train_ema = self.train_ema
        train_ali = self.train_ali
        print("This is length of phoneme in pre_process", len(train_ali))
        phoneme=[]
        new_phoneme=[]
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
            return (train_new_phoneme[index],  train_phoneme_len[index], train_new_ema[index], train_ema_len[index])
        else:
            return (test_new_phoneme[index], test_phoneme_len[index], test_new_ema[index], test_ema_len[index])




class train_ema(torch.utils.data.Dataset):

    def get_mel_text_pair(self, index):
        # separate filename and text
        
        text_padded, input_lengths, mel_padded , output_lengths = pre_process(index)
        return (np.array(text_padded), np.array(input_lengths), np.array(mel_padded) , np.array(output_lengths))

    def __getitem__(self, index):
   
        return self.get_mel_text_pair(index)

    def __len__(self):
        return 400



class test_ema(torch.utils.data.Dataset):

    def get_mel_text_pair(self, index):
        # separate filename and text
        
        text_padded, input_lengths, mel_padded , output_lengths = pre_process(index, train=False)
        return (np.array(text_padded), np.array(input_lengths), np.array(mel_padded) , np.array(output_lengths))


    def __getitem__(self, index):
     
        return self.get_mel_text_pair(index)
    

    def __len__(self):
        return 60
    


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """

        text_padded, input_lengths, mel_padded , output_lengths = ([] for i in range(4))


        for i in range(len(batch)):
            text_padded.append(batch[i][0])
            input_lengths.append(batch[i][1])
            mel_padded.append(batch[i][2])
            output_lengths.append(batch[i][3])
            # print(input_lengths)
        # Right zero-pad all one-hot text sequences to max input length
        id_ph = np.argsort(input_lengths)[::-1]
        id_ema = np.argsort(output_lengths)[::-1]
        
        text_padded = np.array(text_padded)
        text_padded = text_padded[id_ph]
        
        input_lengths = np.array(input_lengths)
        input_lengths = input_lengths[id_ph]

        mel_padded = np.array(mel_padded)
        mel_padded = mel_padded[id_ema]

        output_lengths = np.array(output_lengths)
        output_lengths = output_lengths[id_ema]
        max_input_len = 60

        # text_padded = torch.LongTensor(len(batch), max_input_len)
        # text_padded.zero_()
        # for i in range(len(ids_sorted_decreasing)):
        #     text = batch[ids_sorted_decreasing[i]][0]
        #     text_padded[i, :text.size(0)] = text

        # # Right zero-pad mel-spec
        # num_mels = batch[0][1].size(0)
        # max_target_len = max([x[1].size(1) for x in batch])
        # if max_target_len % self.n_frames_per_step != 0:
        #     max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
        #     assert max_target_len % self.n_frames_per_step == 0

        # # include mel padded and gate padded
        # mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        # mel_padded.zero_()
        # gate_padded = torch.FloatTensor(len(batch), max_target_len)
        # gate_padded.zero_()
        # output_lengths = torch.LongTensor(len(batch))
        # for i in range(len(ids_sorted_decreasing)):
        #     mel = batch[ids_sorted_decreasing[i]][1]
        #     mel_padded[i, :, :mel.size(1)] = mel
        #     gate_padded[i, mel.size(1)-1:] = 1
        #     output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, \
            output_lengths







