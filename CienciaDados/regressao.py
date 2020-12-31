import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

dados = pd.read_csv('insurance.csv')
#Você pode baixar os dados em: https://www.kaggle.com/mirichoi0218/insurance

#Verificando as primeiras 5 linhas do conjunto de dados
print(dados.head())
print(dados.shape)
print(dados.dtypes)

#Separando as variáveis X e Y:

dados = dados[dados['smoker'] == 'yes']
dados =  dados[dados['sex'] == 'female']
X = dados['bmi'].values
Y = dados['charges'].values

#Plotando os dados
plt.scatter(X, Y)
plt.title("Índice de Massa Corporal vs Custo do Seguro")
plt.xlabel("Índice de Massa Corporal")
plt.ylabel("Custo do Seguro (Dólares)")
plt.show()

#Correlação
r = pearsonr(X, Y)

print(f'Índice de correlação: {r}')



print(f'Índice de correlação (Pearson): {r}')

#Separando dados de treino e de teste
x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size=0.3)

#Precisamos redimensionar os dados para fazer a regressão linear
x_train=x_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)
y_test=y_test.reshape(-1,1)
X = X.reshape(-1,1)

#treinando o modelo
reg = LinearRegression()
reg.fit(x_train,y_train)
pred = reg.predict(X) #Previsão dos dados completos (X, y_pred)
pred2 = reg.predict(x_test) #Previsão apenas dos dados de teste (x_test, y_pred2)

plt.scatter(X, Y, color="blue")
plt.plot(X, pred, color="red")
plt.title("Índice de Massa Corporal vs Custo do Seguro (Dados de Teste)")
plt.xlabel("Índice de Massa Corporal da Cliente")
plt.ylabel("Custo do Seguro (Dólares)")

plt.show()

r_squared = r2_score(y_test, pred2)
print(f'Coeficiente r2: {r_squared}')

residual = y_test - pred2

plt.title('Resíduos')
plt.xlabel('Resíduos (Dólar)')
plt.ylabel('Frequência Absoluta')
plt.hist(residual, rwidth=0.9)
plt.show()