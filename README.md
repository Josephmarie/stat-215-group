# stat-215-group


Live colab is here: https://colab.research.google.com/drive/1gmnwWN0POHAz-zj5zxxbLCiFHuW8ZzJr?usp=sharing
overleaf: 
Meeting: Friday 11am to 12pm

Bogdan: neural network 
Joseph: Logistc regression 
Anqi: Random forest 
Finn: autoencoder 

## Data Splits
- train: image1 
- validation: image2 
- test: image3

Structure: 
code:
-EDA
-Modeling


## Use the following code to load data and rename the column: 
# import packages and load the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import os
from pyreadr import read_r

image_1 = pd.read_csv("../../data/image_data/image1.txt", delim_whitespace=True, header=None)
image_2 = pd.read_csv("../../data/image_data/image2.txt", delim_whitespace=True, header=None)
image_3 = pd.read_csv("../../data/image_data/image3.txt", delim_whitespace=True, header=None)

column_names = ['y_coor', 'x_coor', 'expert_label', 'NDAI', 'SD', 'CORR', 'Radiance_angle_DF','Radiance_angle_CF','Radiance_angle_BF','Radiance_angle_AF', 'Radiance_angle_AN'] 
image_1.columns = column_names
image_2.columns = column_names
image_3.columns = column_names
