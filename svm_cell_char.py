import pandas as pd
import numpy as np
import pylab as pl
import scipy.optimize as opt
from sklearn import preprocessing as pproc
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
cell=pd.read_csv('cell_samples.csv')
print(cell.head())
