import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest

dados = pd.read_csv('insurance.csv')
#Você pode baixar os dados em: https://www.kaggle.com/mirichoi0218/insurance


#Testando a normalidade dos dados (A distribuição dos valores observados de bmi é normal?)

dados = dados[dados['sex'] == 'female'] #Analisando apenas os individuos do sexo feminino

plt.figure(figsize=(10,7), dpi= 80)
sns.distplot(dados['bmi'], color="dodgerblue", label="Compact") #Distribuição de frequência dos indice de massa corporal

plt.show()

#vamos verificar o comportamento da distribuição ln(Y) - Distribuição lognormal

log_Y = np.log(dados['bmi'])

plt.figure(figsize=(10,7), dpi= 80)
sns.distplot(log_Y, color="dodgerblue", label="Compact")

plt.show()

alpha = 0.01
k2, p = normaltest(dados['bmi']) #Calculando as estatísticas de teste

#Hipotese nula: Os dados seguem distribuição normal
#Hipotese alternativa: Os dados não seguem distribuição normal

if p < alpha:
    print("A Hipótese Nula pode ser rejeitada")
else:
    print("A hipótese nula não pode ser rejeitada")