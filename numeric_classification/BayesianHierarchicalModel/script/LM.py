import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom, logistic
import pymc3 as pm

# pyplot init
plt.clf()
plt.grid()

# Data of how many seeds of virtual plants sprout
# Every plant have 10 ovules but not every ovule sprouts
df = pd.read_csv('../input/seeds.csv')
del df['plant.ID']

alpha = df['alpha']
y = df['y']
y_max = 10
len_data = len(df)

#------------------------------------------------
# Visualize distribution of y
y_rng = np.arange(y_max+1)
y_histogram = np.histogram(y,bins=y_max+1)[0]
plt.xlabel('y') 
plt.ylabel('freq')
plt.scatter(y_rng, y_histogram)
plt.savefig('../output/y_frequency.png')

# print(df.describe())
# print(df['y'].value_counts())

#------------------------------------------------
# Try to predict how many seeds of virtual plants sprout "y_pred"
#    w/ Binomial Distribution
# p(y|q) = Bin(y|10,q) = 10_C_y q^y (1-q)^10

# Maximum likelihood estimation
q_pred = sum(y) / y_max / len_data 

# Thus, p = 100 Bin(y|10,q_pred)
rv = binom(y_max, q_pred)
y_pred = [len_data * rv.pmf(k) for k in y_rng]

plt.scatter(y_rng, y_histogram, s=len_data, label='observed data')
plt.plot(y_rng, y_pred)
plt.scatter(y_rng, y_pred, s=len_data, label='predicted data')
plt.savefig('../output/y_predial.png')

#------------------------------------------------
# Define bayesian hierarchial model
# WIP. Belows are pseudo code.
beta = Normal('beta', mu=0, tau=1.0e-2)
tau = Gamma('tau', alpha=1.0e-02, beta=1.0e-02)
alpha = Normal('alpha', mu=0, tau=tau , shape=len_data)
ymu = Sigmoid(beta + alpha)
y = Binomial('y', n=y_max, p=ymu, observed=y)
