# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 13:23:04 2021

@author: prakh
"""
import pandas as pd
from sklearn.utils import shuffle


train_df = pd.read_csv("paraphrasing_dataset.csv")
shuffle_train_df = shuffle(train_df).reset_index(drop=True)

# saving the dataframe
shuffle_train_df.to_csv('shuffled_paraphrasing_dataset.csv', index=False)