import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz


url = 'https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv'
dados = pd.read_csv(url)

change = {'no':0, 'yes':1}
dados.sold = dados.sold.map(change)


ano = datetime.today().year
dados['model_age'] = ano - dados.model_year
dados['km_per_year'] = dados.mileage_per_year * 1.60934
dados = dados.drop(columns=['Unnamed: 0', 'mileage_per_year','model_year'])
print(dados.head())

x = dados[['price','model_age','km_per_year']]
y = dados[['sold']]

SEED = 5
np.random.seed(SEED)
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25,
                                                         stratify = y)
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)

dummy = DummyClassifier()
dummy.fit(treino_x, treino_y)
previsoes = dummy.score(teste_x, teste_y)
print(f"Previsão dummy de {previsoes*100}%")
#acuracia = accuracy_score(teste_y, previsoes) * 100
#print("A acurácia do dummy foi %.2f%%" % acuracia)


raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25,
                                                         stratify = y)

scaler = StandardScaler()
scaler.fit(raw_treino_x)
treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

modelo = SVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)
acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia do SVC com scaler foi %.2f%%" % acuracia)

#decision tree
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25,
                                                         stratify = y)



modelo = DecisionTreeClassifier(max_depth=3)
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)
acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia da arvore de decisão com scaler foi %.2f%%" % acuracia)
dot_data = export_graphviz(modelo, out_file=None, feature_names=x.columns,filled=True, class_names=['No', 'Yes'])
grafico = graphviz.Source(dot_data)
#grafico.format = 'svg'
grafico.render(view=True)
