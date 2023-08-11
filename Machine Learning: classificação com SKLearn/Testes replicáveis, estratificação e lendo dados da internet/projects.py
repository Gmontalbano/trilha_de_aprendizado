import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
seed = 20
debug = False

url = ('https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv')
df = pd.read_csv(url)
change = {0:1, 1:0}
df['finished'] = df.unfinished.map(change)
print(df)

sns.scatterplot(x='expected_hours',y='price',hue='finished', data=df)
plt.savefig('scatterplot.png')  # You can use a different file format (e.g., .jpg, .svg) if needed

x = df[['expected_hours', 'price']]
y = df['finished']

SEED = 20

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y,
                                                         random_state = SEED, test_size = 0.25,
                                                         stratify = y)
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)
