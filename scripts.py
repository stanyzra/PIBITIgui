#import importar_arquivo
import gerarArqDiss
import numpy as np
import os, sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#import gerar_arq_diss

def knn(train_feat = [], train_label = [], test_feat = [], test_label = []):
    
    clf = KNeighborsClassifier(n_neighbors = 1)
    
    clf.fit(train_feat, train_label)
    
    '''
    Depois de criar um modelo de aprendizagem de máquima ele vai tentar classificar
    as seguites amostras...
    '''
    
    # Classifica cada amostra do seguinte vetor
    predict = clf.predict(test_feat)
    
    # Ele retorna a label que acha que é para cada uma das imagens
    #print("Predict: ", predict)
    #print("Rotulos de teste: ", test_label)
    
    #print(cont)
    
    a = 0
    acertos = 0
    
    '''
    Enquanto a for menor que o tamanho do vetor predict, os valores de certa 
    posição do vetor predict e do test_label serão comparados e, se forem iguais, 
    será mostrado na tela a posição do valor no vetor predict e a variável acertos 
    irá ser incrementada e mostrada junto com a porcentagem de acerto total.
    '''
    
    while(a < len(predict)):
        
        if(predict[a] == test_label[a]):
            #print("Posição do acerto do vetor: ", a) 
            acertos += 1 
          
        a += 1
        
    print("Número de acertos: ", acertos)
    sys.stdout.flush()
    print("Porcentagem de acerto: ", ((acertos/len(predict))*100))
    sys.stdout.flush()

def rf(train_feat = [], train_label = [], test_feat = [], test_label = []):
    
    clf = RandomForestClassifier(n_estimators=4000, criterion="entropy", verbose=1, oob_score = True, n_jobs = -1, random_state=0)
    
    clf.fit(train_feat, train_label)
    
    '''
    Depois de criar um modelo de aprendizagem de máquima ele vai tentar classificar
    as seguites amostras...
    '''
    
    # Classifica cada amostra do seguinte vetor
    predict = clf.predict(test_feat)
    # Ele retorna a label que acha que é para cada uma das imagens
    #print("Predict: ", predict)
    #print("Rotulos de teste: ", test_label)
    
    #print(cont)
    
    a = 0
    acertos = 0
    
    '''
    Enquanto a for menor que o tamanho do vetor predict, os valores de certa 
    posição do vetor predict e do test_label serão comparados e, se forem iguais, 
    será mostrado na tela a posição do valor no vetor predict e a variável acertos 
    irá ser incrementada e mostrada junto com a porcentagem de acerto total.
    '''
    
    while(a < len(predict)):
        
        if(predict[a] == test_label[a]):
            #print("Posição do acerto do vetor: ", a) 
            acertos += 1 
          
        a += 1
        
    print("Número de acertos: ", acertos)
    sys.stdout.flush()
    print("Porcentagem de acerto: ", ((acertos/len(predict))*100))
    sys.stdout.flush()
   
def svm(train_feat = [], train_label = [], test_feat = [], test_label = []):
        
    print("gerando arquivo de treino...")  

    x = np.array(train_feat)
            
    fileTreino = open("vetores/vetTreino.txt","w")
    
    conteudoTreino = x
        
    linhas = len(conteudoTreino)
    colunas = len(conteudoTreino[0])
    cont = 0
       
    
    for i in range(linhas):
        cont = 1
        fileTreino.write("{} ".format(train_label[i]))

        for j in range(colunas):
            fileTreino.write("{}:{} ".format(cont, conteudoTreino[i][j]))
            cont += 1   

        fileTreino.write("\n")
    fileTreino.close()
    
    print("gerando arquivo de teste...")  

    y = np.array(test_feat)
       
    fileTeste = open("vetores/vetTeste.txt","w")
   
    conteudoTeste = y
    
    linhas = len(conteudoTeste)
    colunas = len(conteudoTeste[0])

    num_amostras_teste = 160

    for i in range(linhas):
        cont = 1
        fileTeste.write("{} ".format(test_label[i]))

        for j in range(colunas):

            fileTeste.write("{}:{} ".format(cont, conteudoTeste[i][j]))
            cont += 1   

        if i < num_amostras_teste*num_amostras_teste:
            fileTeste.write("\n")

    fileTeste.close()
    

    # Executa o LIBSVM
    arq_treino = 'vetores/vetTreino.txt'
    arq_teste = 'vetores/vetTeste.txt'
    comando = 'python ./easy.py ../../{} ../../{}'.format(arq_treino, arq_teste)
    print(comando)
    os.chdir('./libsvm-3.24/tools/')
    print("Executando SVM, aguarde...")
    os.system(comando)
    os.chdir('./../../')
    print("SVM Finalizado...")
    
   
def diss(train_feat = [], train_label = [], test_feat = [], test_label = []):
    
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
        #print(i)
        v1 = abs(np.float16(x[i]) - np.float16(x[i+1]))
#        print("v1: {}".format(v1))
        v2 = abs(np.float16(x[i]) - np.float16(x[i+2]))
#        print("v2: {}".format(v2))
        v3 = abs(np.float16(x[i+1]) - np.float16(x[i+2]))
#        print("v3: {}".format(v3))

        vp.append(v1)
        vp.append(v2)
        vp.append(v3)
        
   # print("contador: {}".format(i))
    # vn é o array de vetores negativos
    vn = []
    print("Criando vetores negativos...")
    print("Subtração de i - i+p")
    for i in range(0, n-p):
        #print("Contador negativo: {}".format(i))
        v1 = abs(np.float16(x[i]) - np.float16(x[i+p]))
        #print("n-p: {}".format(n-p))
        #print("v1: {}".format(v1))
        
        vn.append(v1) 
        
   # print("contador: {}".format(i))
    gerarArquivoTreino(vp,vn)
    print("gerando arquivo de treino...")  
    gerarArqDiss.gerar(test_feat)

    # Executa o LIBSVM
    arq_treino = 'vetores/vetor_diss_treino.txt'
    arq_teste = 'vetores/vetor_diss_teste.txt'
    comando = 'python ./easyDiego.py ../../{} ../../{}'.format(arq_treino, arq_teste)
    print(comando)
    os.chdir('./libsvm-3.24/tools/')
    print("Executando SVM, aguarde...")
    os.system(comando)
    os.chdir('./../../')
    print("SVM Finalizado...")
    
    fileDiss = open("libsvm-3.24/tools/vetor_diss_teste.txt.predict","r")
    
    conteudoDiss = fileDiss.readlines()
    fileDiss.close()
    
    # Converte as linhas de string para matriz
    linhas = len(conteudoDiss)
    vetDiss = [] #vetor que separa as caracteristicas em colunas para facilitar as operações
    for i in range(linhas):
        np.array(vetDiss.append(conteudoDiss[i].split()))
    
    
    # Função para encontrar posição do maior valor
    #print(np.argmax(vetDiss[2]))
    
    # Conta as colunas do arquivo: % classe1 classe2 .... classeN
    colunas = len(vetDiss[0])
    # Dados para  calcular a acurácia
    acertos = 0
    # Total de amostras no teste, não o total de vetores de dissimilaridade no teste
    total_amostras = 160
    # Passo é igual o total de amostras
    p = total_amostras
    # Posição correta do vetor positivo
    pos_vetor = 0
    
    # Para verificar se a amostra foi classificada corretamente, faz esse passo
    # para verificar qual a maior probabilidade de ser positivo
    # Posição do vetor que é ele - ele: 1, 162, 323, 484... +161
    for i in range(1, linhas, p):
        
        # vetor que é ele - ele    
        pos_vetor += 1
    
        # Zera a % do vetor que é ele - ele
        vetDiss[pos_vetor] = ['0']*colunas
        
        # Encontrar a posição do vetor que deve ser o maior acerto positivo
        if pos_vetor %2 == 0:
            certo_pos = pos_vetor-1
        else:
            certo_pos = pos_vetor+1
            
        # Variáveis para encontrar onde está o maior acerto positivo
        maior_pos = 0
        maior_taxa = 0
        
        # Percorre os 160 vetores para saber qual é o maior acerto positivo
        # Faz um for com base no intervalo
        for j in range(i, i+total_amostras): #de 1 até 160, pois a linha 0 são as labels
        
            if j == pos_vetor:
                pass
        
            if float(vetDiss[j][1]) > maior_taxa:
                maior_taxa = float(vetDiss[j][1])
                maior_pos = j
                
        #print("Posição do maior {} com taxa {}".format(maior_pos, maior_taxa))
                
        if maior_pos == certo_pos:
            acertos += 1
#            print("acertou")
                
#        print("Vetor atual {}, onde ver o vetor certo {}, Maior posição {}".format(pos_vetor, certo_pos, maior_pos))
                
        pos_vetor += total_amostras
        
    print("Acertos: {}/{} = {}%".format(acertos, total_amostras, (acertos/total_amostras)*100))

#aqui geramos os arquivos .txt do vetor POSITIVO com sua label e sua respectiva característica
    
def gerarArquivoTreino(vetP, vetN):  
    fileVP = open("vetores/vetor_diss_treino.txt","w")

    conteudoVp = vetP
    
    linhas = len(conteudoVp)
    colunas = len(conteudoVp[0])
       
    for i in range(linhas):
        cont = 1
        fileVP.write("0 ")
        
        for j in range(colunas):
            fileVP.write("{}:{} ".format(cont, conteudoVp[i][j]))
            cont += 1   

        fileVP.write("\n")

    fileVP.close()
    
    fileVN = open("vetores/vetor_diss_treino.txt","a")

    conteudoVn = vetN
    
    linhas = len(conteudoVn)
    colunas = len(conteudoVn[0])
           
    for i in range(linhas):
        cont = 1
        fileVN.write("1 ")

        for j in range(colunas):
            fileVN.write("{}:{} ".format(cont, conteudoVn[i][j]))
            cont += 1   

        fileVN.write("\n")

    fileVN.close()
    

def fusao(train_feat = [], train_label = [], test_feat = [], test_label = []):
    
    
#    print("Iniciando a fusão dos arquivos {} e {}...".format(arquivo1, arquivo2))
    print("Iniciando fusão")
    features_to_svm_train_test("features/lbp.txt", 80, 5, 3, 2)

    arquivo = open("libsvm-3.24/tools/fusaoTeste.txt.predict", "r")
    predict1 = arquivo.readlines()
    arquivo.close()
    
    features_to_svm_train_test("features/ssd.txt", 80, 5, 3, 2)

    arquivo = open("libsvm-3.24/tools/fusaoTeste.txt.predict", "r")
    predict2 = arquivo.readlines()
    arquivo.close()

    # Cria os vetores para armazenar a fusão
    vetSoma = []
    vetProduto = []
    vetMax = []

    print("Realizando a fusão e buscando a nova classe de cada amostra...")

    # Percorre o arquivo pulando a primeira linha porque só possui as labels
    for linha in range(1, len(predict1)):

        # Gera o vetor com os predicts somente do 0 até o 79, porque 80 (classes) não conta
        vetPredict1 = predict1[linha].replace('\n', '').split(" ")#[0:num_classes]
        vetPredict2 = predict2[linha].replace('\n', '').split(" ")#[0:num_classes]


        # Mostra a última posição do vetor
        # print("Vet predict:", vetPredictLBP[-1])

        # Faz a fusão
        linhaSom = [float(x) + float(y) for x, y in zip(vetPredict1, vetPredict2)]
        linhaPro = [float(x) * float(y) for x, y in zip(vetPredict1, vetPredict2)]
        linhaMax = [x if float(x) >= float(y) else y for x, y in zip(vetPredict1, vetPredict2)]

        # Encontra a posição do maior valor, mas desconsidara a primeira coluna porque é a classe (label)      
        # Se o label começa em 0 deixe como está. Se começar em 1 faça a soma +1
        linhaSom[0] = np.argmax(np.array(linhaSom[1:].copy()))  # + 1
        linhaPro[0] = np.argmax(np.array(linhaPro[1:].copy()))  # + 1
        linhaMax[0] = np.argmax(np.array(linhaMax[1:].copy()))  # + 1

        # Adiciona a linha que foi fundida ao vetor com todos os valores
        vetSoma.append(linhaSom)
        vetProduto.append(linhaPro)
        vetMax.append(linhaMax)

    print("Fusão realizada. Criando arquivos com resultados...")

    # Criar os arquivos do zero no modo escrita
    arquivoSoma = open("predicts/predictSoma.txt", "w")
    arquivoProduto = open("predicts/predictProduto.txt", "w")
    arquivoMax = open("predicts/predictMax.txt", "w")

    # Para cada linha, escreve no arquivo aberto
    for lin in range(len(vetSoma)):
        linhaSoma = str(vetSoma[lin]).replace("[", "").replace("]", "").replace(",", "")
        linhaProduto = str(vetProduto[lin]).replace("[", "").replace("]", "").replace(",", "")
        linhaMax = str(vetMax[lin]).replace("[", "").replace("]", "").replace(",", "")

        arquivoSoma.write(linhaSoma.replace("'", "")+"\n")
        arquivoProduto.write(linhaProduto.replace("'", "")+"\n")
        arquivoMax.write(linhaMax.replace("'", "")+"\n")

    # Fecha os arquivos gerados
    arquivoSoma.close()
    arquivoProduto.close()
    arquivoMax.close()
    
    ler_predict("predicts/predictSoma.txt", 2)
    ler_predict("predicts/predictProduto.txt", 2)
    ler_predict("predicts/predictMax.txt", 2)
    
def ler_predict(nome_arquivo, amostras_por_classe, classe_inicial=0):

    print("Abrindo um predict para leitura...")

    # Abre e lê o arquivo
    arquivo = open(nome_arquivo, "r")
    conteudo = arquivo.readlines()
    arquivo.close()

    # Contador de amostras
    cont = 0

    # Contagem de acertos
    acertos = 0

    # Contador de classe pra ver se acertou
    classe_atual = classe_inicial

    # Verifica se tem a linha das labels
    if "labels" in conteudo[0]:
        inicio = 1
    else:
        inicio = 0
    
    # Percorre as linhas do arquivo
    for linha in range(inicio, len(conteudo)):

        # Verifica se a linha não está vazia
        if conteudo[linha].strip() == "" or conteudo[linha].strip() == "\n":
            continue

        # Conta uma amostra
        cont += 1

        # Transforma a linha do predict em vetor
        predicao = conteudo[linha].split()

        # O primeiro valor é a classe
        classe = int(predicao[0])

        # Se a classe (int) da amostra for igual a classe atual da contagem, conta 1 acerto
        if classe == classe_atual:
            acertos += 1

        # Se a contagem atual for múltipla da quantidade de amostras por classe, incrementa a classe
        if cont % amostras_por_classe == 0:
            classe_atual += 1
    
    acuracia = float(acertos / cont)*100

    print("\nResultado do arquivo {}:".format(nome_arquivo))
    print("Total de amostras: {}".format(cont))
    print("Total de acertos: {}".format(acertos))
    print("Total de acertos: {}/{} equivalente a {:.3f}%".format(acertos, cont, acuracia))
    
def features_to_svm_train_test(data, num_classes, amostras_por_classe, qtde_treino, qtde_teste):

    print("Iniciando a separação das amostras em treino e teste...")

    # Abre o arquivo, lê o conteúdo para um vetor e fecha
    arquivo = open(data, "r")
    conteudo = arquivo.readlines()
    arquivo.close()

    # Controla o ID da amostra
    classe_amostra = 0
    
    # Conta as amostras por classe
    cont_por_classe = 0

    # Vetor para treino e teste
    treino = []
    teste = []
    treino_classes = []
    teste_classes = []


    for linha in range(len(conteudo)):

        # Verifica se a linha não está vazia
        if conteudo[linha].strip() == "" or conteudo[linha].strip() == "\n":
            continue

        # Conta 1 amostra
        cont_por_classe += 1

        # Troca a classe da amostra
        if linha > 0 and linha % amostras_por_classe == 0:
            cont_por_classe = 1
            classe_amostra += 1

        # Usa no treinamento se o cont for até a quantidade de amostras para treino
        if cont_por_classe <= qtde_treino:
            treino.append(conteudo[linha])
            treino_classes.append(classe_amostra)
        # A amostra vai para para o conjunto de teste
        elif cont_por_classe <= qtde_teste + qtde_treino:
            teste.append(conteudo[linha])
            teste_classes.append(classe_amostra)
        # Se ainda sim sobrar amostras, pula essa amostra
        else:
            continue

        # Fim da separação das amostras...
    
    print("Criando arquivo de treino com {} amostra(s) para cada uma das {} classe(s)".format(qtde_treino, num_classes))
    # Abre o arquivo de saída de treino
    arquivo_treino = open("vetores/fusaoTreino.txt", "w")

    # Para cada amostra no treino...
    for i in range(len(treino)):

        # Escreve no arquivo a classe dessa amostra "i"
        arquivo_treino.write("{} ".format(treino_classes[i]))

        # Transforma a linha inteira que é um str em um vetor, quebrando nos espaços
        caracteristicas = treino[i].split()

        # Enumera as características
        enum = 1

        # Para cada característica..
        for c in caracteristicas:

            # Escreve "n:característica" com espaço no final 
            arquivo_treino.write("{}:{} ".format(enum, c))

            enum += 1
        
        # Quebra linha
        arquivo_treino.write("\n")
    
    # Fecha o arquivo
    arquivo_treino.close()

    print("Criando arquivo de teste com {} amostra(s) para cada uma das {} classe(s)".format(qtde_teste, num_classes))
    # Abre o arquivo de saída de treino
    arquivo_teste = open("vetores/fusaoTeste.txt", "w")

    # Para cada amostra no teste...
    for i in range(len(teste)):

        # Escreve no arquivo a classe dessa amostra "i"
        arquivo_teste.write("{} ".format(teste_classes[i]))

        # Transforma a linha inteira que é um str em um vetor, quebrando nos espaços
        caracteristicas = teste[i].split()

        # Enumera as características
        enum = 1

        # Para cada característica..
        for c in caracteristicas:

            # Escreve "n:característica" com espaço no final
            arquivo_teste.write("{}:{} ".format(enum, c))

            enum += 1

        # Quebra linha
        arquivo_teste.write("\n")

    # Fecha o arquivo
    arquivo_teste.close()
    
    # Executa o LIBSVM
    arq_treino = 'vetores/fusaoTreino.txt'
    arq_teste = 'vetores/fusaoTeste.txt'
    comando = 'python ./easyDiego.py ../../{} ../../{}'.format(arq_treino, arq_teste)
    print(comando)
    os.chdir('./libsvm-3.24/tools/')
    print("Executando SVM, aguarde...")
    os.system(comando)
    os.chdir('./../../')
    print("SVM Finalizado...")
    