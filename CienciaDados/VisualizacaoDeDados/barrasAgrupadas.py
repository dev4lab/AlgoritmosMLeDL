import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

dados = pd.read_csv('insurance.csv')
#Você pode baixar este arquivo em: https://www.kaggle.com/mirichoi0218/insurance

grupos = ['Fumantes', 'Não Fumantes']
total_fumantes = dados[dados['smoker'] == 'yes']
total_nao_fumantes = dados[dados['smoker'] == 'no']

homens_n_fumantes = total_nao_fumantes[total_nao_fumantes['sex'] == 'male']
mulheres_n_fumantes = total_nao_fumantes[total_nao_fumantes['sex'] == 'female']

homens_fumantes = total_fumantes[total_fumantes['sex']=='male']
mulheres_fumantes = total_fumantes[total_fumantes['sex']=='female']

homens = [homens_fumantes.shape[0], homens_n_fumantes.shape[0]]
mulheres = [mulheres_fumantes.shape[0], mulheres_n_fumantes.shape[0]]

x = np.arange(len(grupos))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, homens, width, label='Homens')
rects2 = ax.bar(x + width/2, mulheres, width, label='Mulheres')

ax.set_ylabel('Número de pessoas')
ax.set_title('Quantidade de Clientes Fumantes e Não Fumantes')
ax.set_xticks(x)
ax.set_xticklabels(grupos)
ax.legend()

plt.show()