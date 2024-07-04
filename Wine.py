# Base de dados utilizada:

# Wine:

# Descrição: Dados químicos de diferentes vinhos.
# Aplicação: Classificação de tipos de vinho.
# Tamanho: 178 instâncias.
# Atributos: 13 atributos (características químicas).

# Importando bibliotecas necessárias e carregando a base de dados
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Dividindo a base em Treinamento e Teste.
X,y =load_wine(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

# Instanciando o classificador NaiveBayes com modelagem dos dados por distribuição normal.
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Executando o classificador com dados de teste e obtendo resultados previstos.
y_pred = gnb.predict(X_test)

# Comparando os resultados previstos pelo classificador com os resultados esperados de acordo com os dados de teste (Total: 89, Numero erros: 2)
soma = (y_test != y_pred).sum()
print(f"Total: {X_test.shape[0]}")
print(f"Numero erros: {soma}")