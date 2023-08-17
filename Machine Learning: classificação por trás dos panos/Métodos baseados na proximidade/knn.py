import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
dados = pd.read_csv('../Customer-Churn.csv')
pd.set_option('display.max_columns', 39)

traducao = {'Sim': 1,
            'Nao': 0}
dadosmodificados = dados[['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn']].replace(traducao)

#transformação pelo get_dummies
dummie_dados = pd.get_dummies(dados.drop(['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn'],
                axis=1))

#junção dos dados trasformados com os que já tinhamos
dados_final = pd.concat([dadosmodificados, dummie_dados], axis=1)

x = dados_final.drop('Churn', axis=1)
y = dados_final['Churn']

norm = StandardScaler()
x_norm = norm.fit_transform(x)
Xmaria = [[0,0,1,1,0,0,39.90,1,0,0,0,1,0,1,0,0,0,0,1,1,1,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,1]]
x_maria_norm = norm.transform(pd.DataFrame(Xmaria, columns= x.columns))
#Euclidiana
a = x_maria_norm
b = x_norm[0]
sub = a-b
square = np.square(sub)
sum = np.sum(square)
print(sum)
print(np.sqrt(sum))

print(10*('-'))

treino_x, teste_x, treino_y, teste_y = train_test_split(x_norm, y, test_size = 0.3, random_state=123)

knn = KNeighborsClassifier(metric='euclidean')
knn.fit(treino_x, treino_y)
predict = knn.predict(teste_x)
print(predict)