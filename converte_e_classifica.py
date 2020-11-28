# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 20:14:10 2020

@author: aleix
"""
import os, sys, re, shutil
import numpy as np

# Lê o arquivo com os índices
# fidIndice = open('indices.txt', 'r')
# conteudo = fidIndice.readline()
# fidIndice.close()

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

#####################################################################
"""
print("Deseja converter quais dados?")
dir_conversao = r"arquivos_treino"
dir_bases = r".\PIBITI"
id_conversao = []
total_dir = len(os.listdir(dir_bases))
bases = os.listdir(dir_bases)
#print(len(os.listdir(dir_extracao)))
for i in range(total_dir):
    print("{} - {}".format(i, bases[i]))
    id_conversao.append(i)
decisao = int(input())

print(bases[decisao])
"""
def converterEClassificar(dir_conversao, base):
    # Verifica se é Windows ou Linux
    dir_svm = r".\libsvm-3.24\tools"

    is_win = (sys.platform == 'win32')
    # Se for windows...
    if is_win:
        #r'C:\Users\aleix\Documents\teste\teste_audio\PIBITI'
    	dir_origem = r"PIBITI"
    # Se for Linux...
    else:
        dir_origem = r"PIBITI"
    	
        
    dir_conversao += r'\caracteristicas-'+base
    extensao_txt = ".txt"
    nome_txt = [_ for _ in os.listdir(dir_conversao) if _.endswith(extensao_txt)]
    
    
    file_conversao = open(r"{}".format(dir_conversao) + r"\{}".format(nome_txt[0]), 'r')
    
    conteudo = file_conversao.readlines()
    conteudo = natural_sort(conteudo)
    
    file_conversao.close()
    
    pessoa = 1
    cont = 1
    
    train_feat = []
    train_label = []
    
    for linha in range(len(conteudo)):
            # A cada 5 amostras troca-se a pessoa
            if linha > 0 and linha % 5 == 0:
                pessoa += 1
                cont = 1
            # Usa no treinamento se o cont for até a quantidade de amostras para treino
            np.array(train_feat.append(conteudo[linha].split()))
            train_label.append(str(pessoa))
            cont += 1
            
    
    x = np.array(train_feat)
    treino = dir_conversao+r'\convertido-'+base+'.svm'
    fileTreino = open(treino,"w")
    
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
    
    os.chdir(dir_svm)
    
    
    # Executa o libsvm
    comando_easy = 'python easy_sistema.py ..\..\%s\%s' %(dir_conversao,'convertido-'+base+'.svm')
    os.system(comando_easy)
    print(comando_easy)
    #print comando_easy
    
    
    dir_destino = r"resultados"
    # Copia os models para a pasta dos folds
    #comando_copy = 'cp %s.{model,range,scale,scale.out,scale.png} %s' % (treino,dir_origem)
    comando_copy = 'move /Y %s.* ..\..\%s' % ('convertido-'+base+'.svm',dir_destino)
    
    os.system(comando_copy)
    print(comando_copy)
    #print comando_copy
    
    os.chdir(r"..\..\..\pibiti")
