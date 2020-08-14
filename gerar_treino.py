import numpy as np
import gerar_teste

fileh = open("features/lbp.txt", "r")
    
conteudo = fileh.readlines()

fileh.close()

# Agora que vocês sabem que a variável "conteúdo" é um vetor,
# faça um laço de repetição para mostrar na tela os nomes do arquivo

# Controla o ID da pessoa
pessoa = 1
# Quantas amostras das 5 serão utilizadas no treinamento
amostras_treino = 3
# Conta as amostras de cada pessoa
cont = 1

# Vetor para treinamento do modelo e rótulos
train_feat = []
# Criação das labens (rótulos/classes) para o classificador
train_label = []

''' Será usado para o classificador '''
# Vetor para teste de predict
test_feat = []
# Criação das labens (rótulos/classes) para o classificador
test_label = []

print("Separando as amostras de treino e de teste...")
for linha in range(len(conteudo)):
    
    # A cada 5 amostras troca-se a pessoa
    if linha > 0 and linha % 5 == 0:
        pessoa += 1
        cont = 1

    # Usa no treinamento se o cont for até a quantidade de amostras para treino
    if cont <= amostras_treino:
        np.array(train_feat.append(conteudo[linha].split()))
        train_label.append(str(pessoa))
    # Se não, joga a amostra para o conjunto de teste
    else:
        test_feat.append(conteudo[linha].split())
        test_label.append(str(pessoa))
    # Conta mais uma amostra para a próxima rodada do for
    cont += 1

# Número total de classes
num_classes = 80
# n é o numero de amostras do treinamento
n = 3 * num_classes
# p é a quantidade de amostras por pessoa
p = 3
# vp é o array de vetores positivos
vp = []
x = np.array(train_feat)
#x = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10], [11, 11, 11], [12, 12, 12]]
print("Criando vetores positivos...")
print("Subtração de i - i+1")
print("Subtração de i - i+2")
print("Subtração de i+1 - i+2")

for i in range(0, n, p):
    #print("Contador positivo: {}".format(i))
    print(i)
    v1 = abs(np.float16(x[i]) - np.float16(x[i+1]))
    print("v1: {}".format(v1))
    v2 = abs(np.float16(x[i]) - np.float16(x[i+2]))
    print("v2: {}".format(v2))
    v3 = abs(np.float16(x[i+1]) - np.float16(x[i+2]))
    print("v3: {}".format(v3))
    
    vp.append(v1)
    vp.append(v2)
    vp.append(v3)
    
print("contador: {}".format(i))
# vn é o array de vetores negativos
vn = []
print("Criando vetores negativos...")
print("Subtração de i - i+p")
for i in range(0, n-p):
    print("Contador negativo: {}".format(i))
    v1 = abs(np.float16(x[i]) - np.float16(x[i+p]))
    print("n-p: {}".format(n-p))
    print("v1: {}".format(v1))
    
    vn.append(v1) 
    
print("contador: {}".format(i))
#aqui geramos os arquivos .txt do vetor POSITIVO com sua label e sua respectiva característica
    
def gerarArquivoTreino():  
    fileVP = open("vetTreino.txt","w")

    conteudoVp = vp
    
    linhas = len(conteudoVp)
    colunas = len(conteudoVp[0])
       
    for i in range(linhas):
        
        for j in range(colunas):
            fileVP.write("{} ".format(conteudoVp[i][j]))

        fileVP.write("\n")

    fileVP.close()
    
    fileVN = open("vetTreino.txt","a")

    conteudoVn = vn
    
    linhas = len(conteudoVn)
    colunas = len(conteudoVn[0])
           
    for i in range(linhas):

        for j in range(colunas):
            fileVN.write("{} ".format(conteudoVn[i][j]))

        fileVN.write("\n")

    fileVN.close()

gerarArquivoTreino()
print("gerando arquivo de treino...")  
gerar_teste.gerar(test_feat)
print("fim")