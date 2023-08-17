import pandas as pd
dados = pd.read_csv('../Customer-Churn.csv')
pd.set_option('display.max_columns', 39)
#print(dados.shape)
#print(dados.head())
#Numericas x Categóricas
traducao = {'Sim': 1,
            'Nao': 0}
dadosmodificados = dados[['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn']].replace(traducao)

#transformação pelo get_dummies
dummie_dados = pd.get_dummies(dados.drop(['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn'],
                axis=1))

#junção dos dados trasformados com os que já tinhamos
dados_final = pd.concat([dadosmodificados, dummie_dados], axis=1)

print(dados_final)
