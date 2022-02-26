# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 17:33:33 2021

@author: Nadian
"""

import os
# import re
# import shutil
# import string
# import tensorflow as tf




Yes_folder   = 'TextClassification/Yes'
No_folder    =  'TextClassification/No'


with open('YesNoDataset.txt','r') as file:
  TextData = file.readlines()

dataset = []
for line in TextData:
    #clean line
    #separate 
    smpl_lbl= line.split('.')[1].split(',')
    smpl_lbl[1] = smpl_lbl[1].replace("\n","") 
    smpl_lbl[1] = smpl_lbl[1].replace("\'","")
    smpl_lbl[1] = smpl_lbl[1].replace('"','')
    smpl_lbl[1] = smpl_lbl[1].replace(' ','')

    dataset.append({'label':smpl_lbl[0],'sample':smpl_lbl[1]})
    
    
document_Counter = 0
for item in dataset:
  document_Counter = document_Counter + 1
  # print(document_Counter)
  if item['label']=='yes':
    file_name = 'Yes'+ str(document_Counter)+'.txt'
    path = os.path.join(Yes_folder,file_name)
    with open(path,'w') as file:
      file.write(item['sample'])
  elif item['label']=='no':
    file_name = 'No'+ str(document_Counter)+'.txt'
    path = os.path.join(No_folder,file_name)
    with open(path,'w') as file:
      file.write(item['sample'])
  # elif item['label'] =='maybe':
  #   file_name = 'Maybe'+ str(document_Counter)+'.txt'
  #   path = os.path.join(Maybe_folder,file_name)
  #   with open(path,'w') as file:
  #     file.write(item['sample'])
  
