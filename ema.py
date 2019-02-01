'''
Purpose of making this class is to load EMA data into TacoTron 2
'''

import os
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import HTK

def read_data():
    EmaDir = '../../DataBase/Ankur_C/Neutral/EmaClean/'
    AliDir = '../../DataBase/Ankur_C/Neutral/ForceAlign/'
    emafiles = sorted(os.listdir(self.EmaDir))
    alifiles = sorted(os.listdir(self.AliDir))
    train_ema = [loadmat(EmaDir+idx) for idx in tqdm_notebook(emafiles)]
    train_ali = [pd.read_csv(AliDir+idx,header=None) for idx in tqdm_notebook(alifiles)]

    return (train_ema, train_ali)


def pre_process(train_ema, train_ali, train = True):
        # train_ema = self.train_ema
        # train_ali = self.train_ali

        phoneme=[]
        new_phoneme=[]
        set_phoneme=[]
        time_phoneme=[]
        time_sil=[]

        for i in trange(460):
            phoneme.append(train_ali[i][0].map(lambda x:x.split()))
            
        for i in trange(460):
            new_phoneme.append(list(phoneme[i][1:-1].map(lambda x: x[2])))
            time_phoneme.append(list(phoneme[i].map(lambda x: [float(x[0]),float(x[1])])))
            set_phoneme.extend(list(phoneme[i][1:-1].map(lambda x: x[2])))

        begin=0
        end=0
        for i in trange(460):
            begin=int(float(time_phoneme[i][0][1])*100)
            end=int(float(time_phoneme[i][-1][0])*100)
            time_sil.append([begin,end])
            
        for i in trange(460):
            time_phoneme[i]=time_phoneme[i][1:-1]
            
        for i in trange(460):
            time_phoneme[i]=(np.multiply(time_phoneme[i],100))
            
        for i in trange(len(time_phoneme)):
            time_phoneme[i]=list(map(lambda x:int(round(x[1]-x[0])),time_phoneme[i]))

        EOS=['</s>']
        SOS=['<s>']
        zero='0'
        set_phoneme.extend(EOS)
        set_phoneme.extend(SOS)
        set_phoneme.extend(zero)
        EOS='</s>'
        SOS='<s>'

        word_to_int={}
        int_to_word={}

        word_to_int=dict((y,x) for x,y in enumerate(set_phoneme))
        int_to_word=dict((x,y) for x,y in enumerate(set_phoneme))

        for i in trange(len(new_phoneme)):
            for j in range(len(new_phoneme[i])):
                new_phoneme[i][j]=word_to_int[new_phoneme[i][j]]

        copy_phoneme = deepcopy(new_phoneme)     

        set_phoneme=set(set_phoneme)
        maxlen=max([len(new_phoneme[i]) for i in range(len(new_phoneme))])# largest length of a sentence is 62, and it's 331'th  item, which mean"

        for i in trange(np.shape(new_phoneme)[0]):
            for _ in range(maxlen-np.shape(new_phoneme[i])[0]):
                new_phoneme[i].append(word_to_int[EOS])

        embed_phoneme = np.empty((len(set_phoneme),100), dtype=np.float32)

        for i in trange(len(set_phoneme)):
            for j in range(100):
                embed_phoneme[i][j] = np.random.normal(loc=0, scale=1, size=1)[0]
        
        ema=[]
        new_ema=[]

        for i in trange(460):
            ema.append(train_ema[i]['EmaData'])

        for i in trange(460):
            EMA_temp=ema[i]
            EMA_temp=np.transpose(EMA_temp)# time X 18
            Ema_temp2=np.delete(EMA_temp, [4,5,6,7,10,11],1) # time X 12 Supposedly these dimensions of information contains most data.
            MeanOfData=np.mean(Ema_temp2,axis=0) 
            Ema_temp2-=MeanOfData
            C=np.sqrt(np.mean(np.square(Ema_temp2),axis=0))
            Ema=Ema_temp2#np.divide(Ema_temp2,C) # Mean remov & var normailized
            [aE,bE]=Ema.shape
            new_ema.append(Ema)

        for i in range(460):
            new_ema[i]=new_ema[i][time_sil[i][0]:time_sil[i][1]]

        for i in range(460):
            new_ema[i]=np.transpose(new_ema[i])

        dec_ema=new_ema.copy()

        max_len_art=max([new_ema[i].shape[1] for i in range(460)]) #Value is 466, which happens to be the same value 
        putt=np.full((12, 1), word_to_int[EOS])
        target=[]


        for i in trange(460):
            for j in range(max_len_art -new_ema[i].shape[1]):
                new_ema[i]=np.concatenate((new_ema[i],putt),axis=1)
            new_ema[i]=np.transpose(new_ema[i])
            target.append(new_ema[i][:])

        xtrain=[]; xtarget=[]; idd=[]
        ttrain=[]; ttarget=[]; tlen = []
        vtrain=[]; vtarget=[]
        sos_put= np.full((12, 1), float(word_to_int[SOS]))
        eos_put= np.full((12, 1), float(word_to_int[EOS]))
        # sos_put = list(np.ones(12) * float(word_to_int[SOS]))


        F=10
        for i in np.arange(0,460):
            if (((i+F)%10)==0):  #Test
                ttarget.append(target[i])
                ttrain.append(new_phoneme[i])
                tlen.append(np.shape(copy_phoneme[i])[0])
                

        #     elif (((i+F+1)%10)==0): #Validation
        #         vtarget.append(target[i])
        #         vtrain.append(new_phoneme[i])
            
            else: # Train
                xtarget.append(target[i])
                xtrain.append(new_phoneme[i])
        #         xlen.append(enc_len[i])
                idd.append(i)
                
                
                
        xtrain  = np.array(xtrain)
        xtarget = np.array(xtarget)
        ttrain = np.array(ttrain)
        ttarget = np.array(ttarget)
        vtrain = np.array(vtrain)
        vtarget = np.array(vtarget)
        # xlen = np.array(xlen)
        idd = np.array(idd)

        dec_len = np.array([])
        enc_len = np.array([])


        for i in range(xtrain.shape[0]):
            dec_len=np.append(dec_len, dec_ema[idd[i]].shape[1] + 2)
            enc_len = np.append(enc_len, np.shape(copy_phoneme[idd[i]])[0])
            woo = np.hstack((sos_put, dec_ema[idd[i]]))
            
            for j in range(max_len_art+2 - dec_ema[idd[i]].shape[1]-1):
                woo = np.hstack((woo, eos_put))
            
            dec_in = np.append(dec_in, np.asarray(np.transpose(woo)))

        dec_in = dec_in.reshape((xtrain.shape[0], -1 , 12))

        if train:
            return [xtrain, xtarget]
        else:
            return [ttrain, ttarget]



class train_ema():
    def __init__(self):
        self.train_ema, self.train_ali = read_data()
        self.audiopath_and_text = pre_process(self.train_ema, self.train_ali)

        
    

    def get_mel_text_pair(self, audiopath_and_text, index):
        # separate filename and text
        phoneme, ema = audiopath_and_text[0][index], audiopath_and_text[1][index]
        return (phoneme, ema)   

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopath_and_text[index])



class test_ema():
    def __init__(self):
        self.test_ema, self.test_ali = read_data()
        self.audiopath_and_text = pre_process(self.test_ema, self.test_ali, train=False)

        
    

    def get_mel_text_pair(self, audiopath_and_text, index):
        # separate filename and text
        phoneme, ema = audiopath_and_text[0][index], audiopath_and_text[1][index]
        return (phoneme, ema)   

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopath_and_text[index])
    

        
            


    

        

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.)







