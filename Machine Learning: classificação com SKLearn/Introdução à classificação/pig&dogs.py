'''
# Modelo de introdução a classificação
## Classificação entra porco e cachorro

#### Features 1-0
- Pelo longo
- Perna curta
- Faz auau
'''

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# Dados para treino
pig1 = [0, 1, 0]
pig2 = [0, 1, 1]
pig3 = [1, 1, 0]

dog1 = [0, 1, 1]
dog2 = [1, 0, 1]
dog3 = [1, 1, 1]
r = {1: 'Pig', 0: 'Dog'}
# pig = 1
# dog = 0
train_x = [pig1, pig2, pig3, dog1, dog2, dog3]
train_y = [1, 1, 1, 0, 0, 0]

model = LinearSVC()

model.fit(train_x, train_y)

doubt = [1, 1, 1]

answer = model.predict([doubt])
print(r[answer[0]])

misterio1 = [1,1,1]
misterio2 = [1,1,0]
misterio3 = [0,1,1]

testes = [misterio1, misterio2, misterio3]
respostas = model.predict(testes)
for a in respostas:
    print(r[a])
testes_classes = [0, 1, 0]

taxa_de_acerto = accuracy_score(testes_classes, respostas)
print("Taxa de acerto", taxa_de_acerto * 100)