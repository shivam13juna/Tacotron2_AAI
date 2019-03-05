#!/usr/bin/env python
# coding: utf-8

# ## Tacotron 2 inference code 
# Edit the variables **checkpoint_path** and **text** to match yours and run the entire code to generate plots of mel outputs, alignments and audio synthesis from the generated mel-spectrogram using Griffin-Lim.

# #### Import libraries and setup matplotlib

# In[1]:
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import IPython.display as ipd

import sys
# sys.path.append('waveglow
                
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
# from layers import TacotronSTFT
# from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
import pickle
from scipy.stats import pearsonr
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def plot_data(data, figsize=(16, 4)):
    
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none')


# #### Setup hparams

# In[3]:


hparams = create_hparams()
hparams.sampling_rate = 22050


# #### Load model from checkpoint

# In[4]:


checkpoint_path = "checkpoints/checkpoint_560000"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.eval()


# In[5]:


with open('new_var/test_new_phoneme', 'rb') as handle:
    test_phoneme= pickle.loads(handle.read())
    
with open('new_var/test_new_ema', 'rb') as handle:
    test_ema= pickle.loads(handle.read())
    
with open('new_var/test_ema_len', 'rb') as handle:
    test_ema_len= pickle.loads(handle.read())
    
test_phoneme = np.array(test_phoneme)
test_ema = np.array(test_ema)
test_ema_len = np.array(test_ema_len)

with open('new_var/train_new_phoneme', 'rb') as handle:
    train_phoneme= pickle.loads(handle.read())
    
with open('new_var/train_new_ema', 'rb') as handle:
    train_ema= pickle.loads(handle.read())
    
    
with open('new_var/train_ema_len', 'rb') as handle:
    train_ema_len= pickle.loads(handle.read())
    

    
train_phoneme = np.array(train_phoneme)
train_ema = np.array(train_ema)




# #### Prepare text input

# In[6]:


# sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(test_phoneme)).cuda().long()


# #### Decode text input and plot results

# In[7]:


mel_outputs, mel_outputs_postnet, alignments = model.inference(sequence)
# plot_data((mel_outputs.data.cpu().numpy()[0],
#            mel_outputs_postnet.data.cpu().numpy()[0],
#            alignments.data.cpu().numpy()[0].T))


# In[8]:


output = mel_outputs.cpu().detach().numpy()
postnet_output = mel_outputs_postnet.cpu().detach().numpy()


# In[9]:


np.shape(postnet_output)


# In[10]:


check = []

for i in test_ema_len:
    check.append(int(i))


# In[11]:


# noo, point= 8, 11

# plt.plot(output[noo,:check[noo],point])
# plt.xlabel('Prediction')
# plt.show()



# plt.plot(test_ema[noo,:check[noo],point])
# plt.xlabel('Actual Output')
# plt.show()

# plt.plot(output[noo,:,point])
# plt.xlabel('Prediction')
# plt.show()



# plt.plot(test_ema[noo,:,point])
# plt.xlabel('Actual Output')
# plt.show()





# print(pearsonr(test_ema[noo,:,11],translation[noo,:,11])[0])


# In[12]:


count=0
avg=0.0
for i in range(test_ema.shape[0]):
    for j in range(test_ema.shape[2]):
        avg+=pearsonr(test_ema[i,:,j],output[i,:,j])[0]
        count+=1
avg=float(avg/count)

print("This is the test_avg with padding: ",avg)


count=0
avg=0.0
for i in range(test_ema.shape[0]):
    for j in range(test_ema.shape[2]):
        avg+=pearsonr(test_ema[i,:check[i],j],output[i,:check[i],j])[0]
        count+=1
avg=float(avg/count)
print("This is the test_avg without padding: ",avg)


# #### Prepare text input

# In[6]:


# sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence=torch.autograd.Variable(
    torch.from_numpy(train_phoneme)).cuda().long()


# #### Decode text input and plot results

# In[7]:


mel_outputs, mel_outputs_postnet, alignments = model.inference(sequence)
# plot_data((mel_outputs.data.cpu().numpy()[0],
#            mel_outputs_postnet.data.cpu().numpy()[0],
#            alignments.data.cpu().numpy()[0].T))


# In[8]:


output = mel_outputs.cpu().detach().numpy()
postnet_output = mel_outputs_postnet.cpu().detach().numpy()


# In[16]:


np.shape(postnet_output)


# In[17]:


check = []

for i in train_ema_len:
    check.append(int(i))


# In[18]:


noo, point= 1, 5

# plt.plot(output[noo,:check[noo],point])
# plt.xlabel('Prediction')
# plt.show()



# plt.plot(train_ema[noo,:check[noo],point])
# plt.xlabel('Actual Output')
# plt.show()


# plt.plot(output[noo,:,point])
# plt.xlabel('Prediction')
# plt.show()



# plt.plot(train_ema[noo,:,point])
# plt.xlabel('Actual Output')
# plt.show()




# print(pearsonr(test_ema[noo,:,11],translation[noo,:,11])[0])


# In[19]:


count=0
avg=0.0
for i in range(train_ema.shape[0]):
    for j in range(train_ema.shape[2]):
        avg+=pearsonr(train_ema[i,:,j],output[i,:,j])[0]
        count+=1
avg=float(avg/count)

print("This is the train_avg with padding: ",avg)


count=0
avg=0.0
for i in range(train_ema.shape[0]):
    for j in range(train_ema.shape[2]):
        avg+=pearsonr(train_ema[i,:check[i],j],output[i,:check[i],j])[0]
        count+=1
avg=float(avg/count)

print("This is the train_avg without padding: ",avg)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




