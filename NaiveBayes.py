import math

class NaiveBayes:
    def __init__(self, dados_iris):
        self.dados_iris = dados_iris
        self.sumarios = {}

    # Para cada categoria da Íris, serão calculados a média e o desvio padrão dos quatro atributos das instâncias.
    def sumarizar_por_classe(self):
        separado = self.separar_por_classe()
        sumarios = {}
        for valor_classe, instancias in separado.items():
            atributos = zip(*instancias)
            medias = [self.calcularMedia(atributo) for atributo in atributos]
            atributos = zip(*instancias)
            desvios = [self.calcularDesvio(atributo) for atributo in atributos]
            mediasDesvios = zip(medias, desvios)
            mediasDesvios = list(mediasDesvios)
            sumarios[valor_classe] = mediasDesvios
        self.sumarios = sumarios

    # As instâncias da Íris serão organizadas em um dicionário que conterá chaves que dizem respeito à cada categoria da Íris e associada a cada chave uma lista de instâncias que dizem respeito a ela.
    def separar_por_classe(self):
        separado = {}
        quantidadeIndices = len(self.dados_iris.classes)
        for i in range(quantidadeIndices):
            sepal_length = self.dados_iris.sepal_length[i]
            sepal_width = self.dados_iris.sepal_width[i]
            petal_length = self.dados_iris.petal_length[i]
            petal_width = self.dados_iris.petal_width[i]
            instancia = (sepal_length, sepal_width, petal_length, petal_width)
            valor_classe = self.dados_iris.classes[i]
            if valor_classe not in separado:
                separado[valor_classe] = []
            separado[valor_classe].append(instancia)
        return separado

    def calcularMedia(self, valores):
        return sum(valores) / float(len(valores))

    def calcularDesvio(self, valores):
        media = self.calcularMedia(valores)
        variancia = sum([pow(x - media, 2) for x in valores]) / float(len(valores) - 1)
        return math.sqrt(variancia)

    # Calcular a probabilidade de encontrar um valor de um atributo x em uma distribuição normal (média e desvio)
    def calcularProbabilidadeValorAtributo(self, x, media, desvio):
        exponent = math.exp(-((x - media) ** 2 / (2 * desvio ** 2)))
        return (1 / (math.sqrt(2 * math.pi) * desvio)) * exponent

    # Calcular a probabilidade de uma instância da Íris pertencer a cada uma das três categorias.
    def calcularProbabilidadeClasse(self, vetorInstancia):
        probabilidades = {}
        for valorClasse, sumariosClasse in self.sumarios.items():
            probabilidades[valorClasse] = 1
            for i in range(len(sumariosClasse)):
                media, desvio = sumariosClasse[i]
                x = vetorInstancia[i]
                probabilidades[valorClasse] *= self.calcularProbabilidadeValorAtributo(x, media, desvio)
        return probabilidades

    # Dos três valores resultantes do método calcularProbabilidadeClasse, é identificado o maior, e a categoria que está associada a esse valor é retornada.
    def predizer(self, vetorInstancia):
        probabilidades = self.calcularProbabilidadeClasse(vetorInstancia)
        melhorRotulo, melhorProbabilidade = None, -1
        for valorClasse, probabilidade in probabilidades.items():
            if melhorRotulo is None or probabilidade > melhorProbabilidade:
                melhorProbabilidade = probabilidade
                melhorRotulo = valorClasse
        return melhorRotulo

    def treinar(self):
        self.sumarizar_por_classe()

    def testar(self, dadosTeste):
        predicoes = []
        for instancia in dadosTeste:
            resultado = self.predizer(instancia)
            predicoes.append(resultado)
        return predicoes