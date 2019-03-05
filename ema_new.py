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
import random
from random import shuffle
# np.random.RandomState(1234)
random.seed(1234)

def pre_process(self, index, train = True):
      
    
# new_ema[i]=np.concatenate((new_ema[i],putt),axis=1)
#Yeah, I know train and one_hot_phoneme are same, yet it's there.


    








        # with open('variables/new_phoneme', 'rb') as handle:
        #     new_phoneme = pickle.loads(handle.read())
        
        # with open('variables/word_to_int', 'rb') as handle:
        #     word_to_int = pickle.loads(handle.read())
    
        # with open('variables/set_phoneme', 'rb') as handle:
        #     set_phoneme = pickle.loads(handle.read())
    
        # with open('variables/time_phoneme', 'rb') as handle:
        #     time_phoneme = pickle.loads(handle.read())

        # with open('variables/time_sil', 'rb') as handle:
        #     time_sil = pickle.loads(handle.read())

        # with open('variables/int_to_word', 'rb') as handle:
        #    int_to_word = pickle.loads(handle.read())

        # with open('variables/max_len_ema', 'rb') as handle:
        #     max_len_ema = pickle.loads(handle.read())
        
        # with open('variables/new_ema', 'rb') as handle:
        #    new_ema = pickle.loads(handle.read())

        # with open('variables/maxlen_phoneme', 'rb') as handle:
        #     maxlen_phoneme = pickle.loads(handle.read())

        # with open('variables/self.train_new_ema', 'rb') as handle:
        #     self.train_new_ema = pickle.loads(handle.read())

        # with open('variables/self.test_new_ema', 'rb') as handle:
        #     self.test_new_ema = pickle.loads(handle.read())

        # with open('variables/self.train_new_phoneme', 'rb') as handle:
        #     self.train_new_phoneme = pickle.loads(handle.read())

        # with open('variables/self.test_new_phoneme', 'rb') as handle:
        #     self.test_new_phoneme = pickle.loads(handle.read())

        # with open('variables/self.train_ema_len', 'rb') as handle:
        #     self.train_ema_len = pickle.loads(handle.read())

        # with open('variables/self.test_ema_len', 'rb') as handle:
        #     self.test_ema_len = pickle.loads(handle.read())

        # with open('variables/self.train_phoneme_len', 'rb') as handle:
        #     self.train_phoneme_len = pickle.loads(handle.read())

        # with open('variables/self.test_phoneme_len', 'rb') as handle:
        #     self.test_phoneme_len = pickle.loads(handle.read())




        if train:
            return (self.train_new_phoneme[index],  self.train_phoneme_len[index], self.train_new_ema[index], self.train_ema_len[index])
        else:
            return (self.test_new_phoneme[index], self.test_phoneme_len[index], self.test_new_ema[index], self.test_ema_len[index])




class train_ema:

    def __init__(self):
        folist = sorted(os.listdir('data'))
        phon = []
        emaa = []
        plen = []
        elen = []
        # max_len_art = 769
        max_len_art = 466
        maxlen = 65
        sped = np.arange(1000, 1011)
        curr_index = 0
        N = 5060
        for i in folist:

            EmaDir= 'data/'+ i + '/EmaClean/'
            AliDir= 'data/' + i + '/ForceAlign/'
        
            emafiles=sorted(os.listdir(EmaDir))
            alifiles=sorted(os.listdir(AliDir))

            train_ema = [loadmat(EmaDir+idx) for idx in emafiles]
            train_ali = [pd.read_csv(AliDir+idx,header=None) for idx in alifiles]

            phoneme=[]
            new_phoneme=[]
            set_phoneme=[]
            time_phoneme=[]
            time_sil=[]

            for i in range(460):
                phoneme.append(train_ali[i][0].map(lambda x:x.split()))
                
            for i in range(460):
                new_phoneme.append(list(phoneme[i][1:-1].map(lambda x: x[2])))
                time_phoneme.append(list(phoneme[i].map(lambda x: [float(x[0]),float(x[1])])))
                set_phoneme.extend(list(phoneme[i][1:-1].map(lambda x: x[2])))

            begin=0
            end=0
            for i in range(460):
                begin=int(float(time_phoneme[i][0][1])*100)
                end=int(float(time_phoneme[i][-1][0])*100)
                time_sil.append([begin,end])
                
            for i in range(460):
                time_phoneme[i]=time_phoneme[i][1:-1]
                
            for i in range(460):
                time_phoneme[i]=(np.multiply(time_phoneme[i],100))
                
            for i in range(len(time_phoneme)):
                time_phoneme[i]=list(map(lambda x:int(round(x[1]-x[0])),time_phoneme[i]))



        
            zero='0'
    
            EOS='</s>'
            SOS='<s>'

    
            # word_to_int={}
            # int_to_word={}

            # word_to_int=dict((y,x) for x,y in enumerate(set_phoneme))
            # int_to_word=dict((x,y) for x,y in enumerate(set_phoneme))
            with open('variables/word_to_int', 'rb') as handle:
                word_to_int = pickle.loads(handle.read())

            for i in range(len(new_phoneme)):
                for j in range(len(new_phoneme[i])):
                    new_phoneme[i][j]=word_to_int[new_phoneme[i][j]]
                new_phoneme[i].append(sped[curr_index])


            curr_index+=1

            copy_phoneme = deepcopy(new_phoneme)   


            maxlen_this=max([len(new_phoneme[i]) for i in range(len(new_phoneme))])# largest length of a sentence is 62, and it's 331'th  item, which mean"
            if maxlen_this > maxlen:
                print("In this case, maxlen_this is greater: ", maxlen_this)

            for i in range(np.shape(new_phoneme)[0]):
                for _ in range(maxlen-np.shape(new_phoneme[i])[0]):
                    new_phoneme[i].append(0)
            phon.extend(new_phoneme)


            ema=[]
            new_ema=[]

            for i in range(460):
                ema.append(train_ema[i]['EmaData'])

            for i in range(460):
                EMA_temp=ema[i]
                EMA_temp=np.transpose(EMA_temp)# time X 18
                Ema_temp2=np.delete(EMA_temp, [4,5,6,7,10,11],1) # time X 12 Supposedly these dimensions of information contains most data.
                MeanOfData=np.mean(Ema_temp2,axis=0) 
                Ema_temp2-=MeanOfData
                C=np.sqrt(np.mean(np.square(Ema_temp2),axis=0))
                Ema=np.divide(Ema_temp2,C) # Mean remov & var normailized
                [aE,bE]=Ema.shape
                new_ema.append(Ema)

            for i in range(460):
                new_ema[i]=new_ema[i][time_sil[i][0]:time_sil[i][1]]

            for i in range(460):
                new_ema[i]=np.transpose(new_ema[i])

            max_len_this=max([new_ema[i].shape[1] for i in range(460)]) #Value is 466, which happens to be the same value 
            if(max_len_this > max_len_art):
                print("Max len of ema is greater and value is:", max_len_this)
            
            putt=np.full((12, 1), 0.0)
            dec_ema=new_ema.copy()


            for i in range(460):
                for j in range(max_len_art -new_ema[i].shape[1]):
                    new_ema[i]=np.concatenate((new_ema[i],putt),axis=1)
                new_ema[i]=np.transpose(new_ema[i])


            emaa.extend(new_ema)
            


            dec_len = np.array([])
            enc_len = np.array([])


            for i in range(460):
                dec_len= np.append(dec_len, dec_ema[i].shape[1] + 2)
                enc_len = np.append(enc_len, np.shape(copy_phoneme[i])[0])
            
            plen.extend(enc_len)
            elen.extend(dec_len)
        
        random.seed(1234)
        shuffle(phon)
        random.seed(1234)
        shuffle(emaa)
        random.seed(1234)
        shuffle(plen)
        random.seed(1234)
        shuffle(elen)
        phon = np.array(phon)
        emaa = np.array(emaa)
        plen = np.array(plen)
        elen = np.array(elen)

        # print("This is phon", np.shape(phon))
        # print("This is ema", np.shape(emaa))
        # print("This is length, phon", np.shape(plen))
        # print("This is length, ema", np.shape(elen))
        # random.seed(1234)
        self.train_phoneme_len = plen[:5000]
        self.test_phoneme_len = plen[5000:]
        self.train_ema_len = elen[0:5000]
        self.test_ema_len = elen[5000:]
        self.train_new_ema = emaa[:5000]
        self.test_new_ema = emaa[5000:]
        self.train_new_phoneme = phon[0:5000]
        self.test_new_phoneme = phon[5000:]

        with open('new_var/train_phoneme_len', 'wb') as handle:  
            pickle.dump(self.train_phoneme_len, handle) 

        with open('new_var/test_phoneme_len', 'wb') as handle:  
            pickle.dump(self.test_phoneme_len, handle) 

        with open('new_var/train_new_ema', 'wb') as handle:  
            pickle.dump(self.train_new_ema, handle) 

        with open('new_var/test_new_ema', 'wb') as handle:  
            pickle.dump(self.test_new_ema, handle) 

        with open('new_var/train_ema_len', 'wb') as handle:  
            pickle.dump(self.train_ema_len, handle) 

        with open('new_var/test_ema_len', 'wb') as handle:  
            pickle.dump(self.test_ema_len, handle) 

        with open('new_var/train_new_phoneme', 'wb') as handle:  
            pickle.dump(self.train_new_phoneme, handle) 

        with open('new_var/test_new_phoneme', 'wb') as handle:  
            pickle.dump(self.test_new_phoneme, handle) 


    def get_mel_text_pair(self, index):
        # separate filename and text
        
        text_padded, input_lengths, mel_padded , output_lengths = pre_process(self, index)
        return (np.array(text_padded), np.array(input_lengths), np.array(mel_padded) , np.array(output_lengths))

    def __getitem__(self, index):
   
        return self.get_mel_text_pair(index)

    def __len__(self):
        return 400



class test_ema:

    def __init__(self):

        folist = sorted(os.listdir('data'))
        phon = []
        emaa = []
        plen = []
        elen = []
        max_len_art = 769
        maxlen = 65
        sped = np.arange(1000, 1011)
        curr_index = 0
        N = 5060
        for i in folist:

            EmaDir= 'data/'+ i + '/EmaClean/'
            AliDir= 'data/' + i + '/ForceAlign/'
        
            emafiles=sorted(os.listdir(EmaDir))
            alifiles=sorted(os.listdir(AliDir))

            train_ema = [loadmat(EmaDir+idx) for idx in emafiles]
            train_ali = [pd.read_csv(AliDir+idx,header=None) for idx in alifiles]

            phoneme=[]
            new_phoneme=[]
            set_phoneme=[]
            time_phoneme=[]
            time_sil=[]

            for i in range(460):
                phoneme.append(train_ali[i][0].map(lambda x:x.split()))
                
            for i in range(460):
                new_phoneme.append(list(phoneme[i][1:-1].map(lambda x: x[2])))
                time_phoneme.append(list(phoneme[i].map(lambda x: [float(x[0]),float(x[1])])))
                set_phoneme.extend(list(phoneme[i][1:-1].map(lambda x: x[2])))

            begin=0
            end=0
            for i in range(460):
                begin=int(float(time_phoneme[i][0][1])*100)
                end=int(float(time_phoneme[i][-1][0])*100)
                time_sil.append([begin,end])
                
            for i in range(460):
                time_phoneme[i]=time_phoneme[i][1:-1]
                
            for i in range(460):
                time_phoneme[i]=(np.multiply(time_phoneme[i],100))
                
            for i in range(len(time_phoneme)):
                time_phoneme[i]=list(map(lambda x:int(round(x[1]-x[0])),time_phoneme[i]))



        
            zero='0'
    
            EOS='</s>'
            SOS='<s>'

    
            # word_to_int={}
            # int_to_word={}

            # word_to_int=dict((y,x) for x,y in enumerate(set_phoneme))
            # int_to_word=dict((x,y) for x,y in enumerate(set_phoneme))
            with open('variables/word_to_int', 'rb') as handle:
                word_to_int = pickle.loads(handle.read())

            for i in range(len(new_phoneme)):
                for j in range(len(new_phoneme[i])):
                    new_phoneme[i][j]=word_to_int[new_phoneme[i][j]]
                new_phoneme[i].append(sped[curr_index])


            curr_index+=1

            copy_phoneme = deepcopy(new_phoneme)   


            maxlen_this=max([len(new_phoneme[i]) for i in range(len(new_phoneme))])# largest length of a sentence is 62, and it's 331'th  item, which mean"
            if maxlen_this > maxlen:
                print("In this case, maxlen_this is greater: ", maxlen_this)

            for i in range(np.shape(new_phoneme)[0]):
                for _ in range(maxlen-np.shape(new_phoneme[i])[0]):
                    new_phoneme[i].append(0)
            phon.extend(new_phoneme)


            ema=[]
            new_ema=[]

            for i in range(460):
                ema.append(train_ema[i]['EmaData'])

            for i in range(460):
                EMA_temp=ema[i]
                EMA_temp=np.transpose(EMA_temp)# time X 18
                Ema_temp2=np.delete(EMA_temp, [4,5,6,7,10,11],1) # time X 12 Supposedly these dimensions of information contains most data.
                MeanOfData=np.mean(Ema_temp2,axis=0) 
                Ema_temp2-=MeanOfData
                C=np.sqrt(np.mean(np.square(Ema_temp2),axis=0))
                Ema=np.divide(Ema_temp2,C) # Mean remov & var normailized
                [aE,bE]=Ema.shape
                new_ema.append(Ema)

            for i in range(460):
                new_ema[i]=new_ema[i][time_sil[i][0]:time_sil[i][1]]

            for i in range(460):
                new_ema[i]=np.transpose(new_ema[i])

            max_len_this=max([new_ema[i].shape[1] for i in range(460)]) #Value is 466, which happens to be the same value 
            if(max_len_this > max_len_art):
                print("Max len of ema is greater and value is:", max_len_this)
            
            putt=np.full((12, 1), 0.0)
            dec_ema=new_ema.copy()


            for i in range(460):
                for j in range(max_len_art -new_ema[i].shape[1]):
                    new_ema[i]=np.concatenate((new_ema[i],putt),axis=1)
                new_ema[i]=np.transpose(new_ema[i])


            emaa.extend(new_ema)
            


            dec_len = np.array([])
            enc_len = np.array([])


            for i in range(460):
                dec_len= np.append(dec_len, dec_ema[i].shape[1] + 2)
                enc_len = np.append(enc_len, np.shape(copy_phoneme[i])[0])
            
            plen.extend(list(enc_len))
            elen.extend(list(dec_len))
        
        random.seed(1234)
        shuffle(phon)
        random.seed(1234)
        shuffle(emaa)
        random.seed(1234)
        shuffle(plen)
        random.seed(1234)
        shuffle(elen)
        # phon = np.array(phon)
        # emaa = np.array(emaa)
        # plen = np.array(plen)
        # elen = np.array(elen)

        # print("This is phon", np.shape(phon))
        # print("This is ema", np.shape(emaa))
        # print("This is length, phon", np.shape(plen))
        # print("This is length, ema", np.shape(elen))
        # random.seed(1234)
        self.train_phoneme_len = plen[:5000]
        self.test_phoneme_len = plen[5000:]
        self.train_ema_len = elen[0:5000]
        self.test_ema_len = elen[5000:]
        self.train_new_ema = emaa[:5000]
        self.test_new_ema = emaa[5000:]
        self.train_new_phoneme = phon[0:5000]
        self.test_new_phoneme = phon[5000:]
        

    def get_mel_text_pair(self, index):
        # separate filename and text
        
        text_padded, input_lengths, mel_padded , output_lengths = pre_process(self, index, train=False)
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







