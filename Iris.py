import csv

class Iris:

    # A classe pode ser instanciada inicializando seus atributos, mas deixando-os sem valores, ou colocando valores neles, se for recebido um arquivo CSV
    def __init__(self, caminho_arquivo=None):
        self.sepal_length = []
        self.sepal_width = []
        self.petal_length = []
        self.petal_width = []
        self.classes = []
        if caminho_arquivo:
            self.importar(caminho_arquivo)

    # Os valores são capturados do CSV através do método importar
    def importar(self, caminho_arquivo):
        try:
            with open(caminho_arquivo, mode='r') as file:
                reader = csv.reader(file)
                next(reader)  # Pular o cabeçalho
                for linha in reader:
                    self.adicionar_dados(float(linha[0]), float(linha[1]), float(linha[2]), float(linha[3]), linha[4])
        except FileNotFoundError as e:
            print(f"Arquivo não encontrado: {e}")

    # Os valores são colocados nos atributos através do método adicionar_dados
    def adicionar_dados(self, sepal_length, sepal_width, petal_length, petal_width, classe):
        self.sepal_length.append(sepal_length)
        self.sepal_width.append(sepal_width)
        self.petal_length.append(petal_length)
        self.petal_width.append(petal_width)
        self.classes.append(classe)