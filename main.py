from Iris import Iris
from NaiveBayes import NaiveBayes
from sklearn.model_selection import train_test_split

# Caminho para o arquivo CSV da base de dados Iris
caminho_arquivo = "Iris.csv"

# Importando a Íris
iris = Iris(caminho_arquivo)

# Criando os vetores de atributos e classes
instancias = list(zip(iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width))
classes = iris.classes

# Dividindo os dados em treinamento e teste (70% treinamento, 30% teste)
instanciasTreino, instanciasTeste, classesTreino, classesTeste = train_test_split(instancias, classes, test_size=0.3, random_state=42, stratify=classes)

# Pegando, dos dados de treino, cada instância e sua respectiva classe.
paresAtributosRotulos = zip(instanciasTreino, classesTreino)

# Pegando as quatro listas (uma para cada atributo) preenchidas com os valores de treino.
atributos = zip(*[(atributo[0], atributo[1], atributo[2], atributo[3], rotulo) for atributo, rotulo in paresAtributosRotulos])

# Instanciando uma Iris vazia e preenchendo-a com os dados de treino.
irisTreino = Iris()
irisTreino.sepal_length, irisTreino.sepal_width, irisTreino.petal_length, irisTreino.petal_width, irisTreino.classes = atributos

# Instanciando um NaiveBayes com os dados de treino.
naive_bayes = NaiveBayes(irisTreino)

# Treinando o modelo com os dados de treino já recebidos.
naive_bayes.treinar()

# Testando o modelo com os dados de teste.
predicoes = naive_bayes.testar(instanciasTeste)

# Calculando a quantidade de erros de predições que o teste gerou em relação aos dados originais utilizados
quantidade_erros = 0
for i in range(len(predicoes)):
    if predicoes[i] != classesTeste[i]:
        quantidade_erros += 1

# Exibindo resultados
print()
print("Total: ", len(predicoes))
print("Erros: ", quantidade_erros)
print()