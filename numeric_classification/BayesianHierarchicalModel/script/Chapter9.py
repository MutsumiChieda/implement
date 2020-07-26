import math
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pyper
import pymc3
pd.set_option('display.max_columns', None) # Show all columns
r = pyper.R()

print(r("""
load("data9.RData")
"""))
data = r.get("d")
data.columns = ['x', 'y']

# 9.1 Applying normal GLM
result = smf.glm(formula='y ~ x', data=data, family=sm.families.Poisson()).fit()
print(result.summary())

pred = np.exp(result.params[0] + result.params[1] * data.x)
plt.plot(data.x, pred)
plt.scatter(data.x, data.y)
plt.savefig('../output/ch9_glm.png')

# 9.3 Non-informative prior
from scipy.stats import norm

x = np.arange(-10.00, 10.01, 0.01)
b1 = [norm.pdf(x_i, 0,   1) for x_i in x]
b2 = [norm.pdf(x_i, 0, 100) for x_i in x]

plt.clf()
plt.plot(x, b1, label='N(0, 1)')
plt.plot(x, b2, label='N(0, 100)')
plt.legend()
plt.xlabel('beta_1, beta_2')
plt.ylabel('p(beta)')
plt.savefig('../output/ch9_non-info.png')

# 9.4 Estimation of posterior of baysian stats model
with pymc3.Model() as model:
    beta1 = pymc3.Normal('beta1', mu=0, sd=100)
    beta2 = pymc3.Normal('beta2', mu=0, sd=100)
    
    lambda_ = np.exp(beta1 + beta2*data['x'].values)
    
    y = pymc3.Poisson('y', mu=lambda_, observed=data['y'].values)