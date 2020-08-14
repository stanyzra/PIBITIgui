import numpy as np
import scripts

def importar():
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
    trf = []
    # Criação das labens (rótulos/classes) para o classificador
    trl = []
    
    ''' Será usado para o classificador '''
    # Vetor para teste de predict
    tef = []
    # Criação das labens (rótulos/classes) para o classificador
    tel = []

    print("Separando as amostras de treino e de teste...")
    for linha in range(len(conteudo)):
        
        # A cada 5 amostras troca-se a pessoa
        if linha > 0 and linha % 5 == 0:
            pessoa += 1
            cont = 1
    
        # Usa no treinamento se o cont for até a quantidade de amostras para treino
        if cont <= amostras_treino:
            np.array(trf.append(conteudo[linha].split()))
            trl.append(str(pessoa))
        # Se não, joga a amostra para o conjunto de teste
        else:
            tef.append(conteudo[linha].split())
            tel.append(str(pessoa))
        # Conta mais uma amostra para a próxima rodada do for
        cont += 1
        
    trf = np.float16(trf)
    trl = np.float16(trl)
    tef = np.float16(tef)
    tel = np.float16(tel)
    
    select = int(input("Selecione a classificacao"))

    if select == 1:
        scripts.knn(trf,trl,tef,tel)
    elif select == 2: 
        scripts.rf(trf,trl,tef,tel)
    elif select == 3:
        scripts.svm(trf,trl,tef,tel)
    elif select == 4:
        scripts.diss(trf,trl,tef,tel)
    elif select == 5:
        scripts.fusao(trf,trl,tef,tel)
    
importar()