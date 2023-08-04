import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
seed = 20
debug = False

url = ('https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw'
       '/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv')
df = pd.read_csv(url)
x = df[['home', 'how_it_works', 'contact']]
y = df['bought']
'''
x, y = Data
stratify = Distribuir de maneira igualitaria os dados
random_state = passando um padrão para sempre utilizar os mesmos dados para teste
test_size = quanto % utilizaremos para teste, o resto fica para treino  
'''
train_x, test_x, train_y, test_y = train_test_split(x, y, stratify=y, random_state=seed, test_size=0.25)
if debug:
    print(df.head())
    # cada elemento tem 99 registros
    print(len(x))
    print(len(y))
'''
train_x = x[:75]
train_y = y[:75]
test_x = x[75:]
test_y = y[75:]
'''
model = LinearSVC()
model.fit(train_x, train_y)
predict = model.predict(test_x)

accuracy = accuracy_score(test_y, predict)

print(f"Accuracy score: {100*accuracy}")
