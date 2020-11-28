# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 20:14:10 2020

@author: aleix
"""

import os, sys, re
import numpy as np
from scipy.io.wavfile import read
import librosa, shutil
import math

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def extrair_audio():

    print("Deseja extair quais dados?")
    dir_extracao = 'PIBITI/'
    id_base = []
    total_dir = len(os.listdir(dir_extracao))
    bases = os.listdir(dir_extracao)
    #print(len(os.listdir(dir_extracao)))
    for i in range(total_dir):
        print("{} - {}".format(i, bases[i]))
        id_base.append(i)
    decisao = int(input())
    
    print(bases[decisao])
    
    
    #####################################################################
    # Diretórios usados para consulta da base e criação dos folds
    dir_origem = 'PIBITI/'
    dir_origem += bases[decisao]+'/'
    # Onde serão armazenados os arquivos gerados (mesma estrutura de divisão)
    dir_destino = 'arquivos_treino/'
    dir_destino += 'caracteristicas-'+bases[decisao]+'/'
    dir_arq = dir_destino+'treino-'+bases[decisao]+'.txt'
    
    if not os.path.exists(dir_destino):
        print("Criando pasta para guardar arquivo de características")
        os.mkdir(dir_destino)
    
    else:
        print("Limpando pasta")
        shutil.rmtree(dir_destino)
        os.mkdir(dir_destino)
    
    # Vai para a pasta do framework
    sys.path.append('./rp_extract')
    # os.chdir('./rp_extract')
    
    from audiofile_read import *
    from rp_extract import rp_extract
    
    
    tipos_feat = {'rh' : False, 'rp' : False, 'ssd': True}
    
    total_pessoas = len(os.listdir(dir_origem))
    cont_amostras = 0
    for pessoa in range(1, total_pessoas+1):
        for amostras in range(1,6):
            cont_amostras += 1
            dir_extracao = dir_origem+'s{}/'.format(pessoa)+'{}.wav'.format(cont_amostras)
            print("Extraindo de {}.wav".format(cont_amostras))
            wavedata, samplerate = librosa.load(dir_extracao, sr=44100)
            feat = rp_extract.rp_extract(wavedata, samplerate, extract_rp=tipos_feat['rp'], extract_ssd=tipos_feat['ssd'], extract_rh=tipos_feat['rh'])
            
            # Verifica cada uma dos três tipos de features acústucas
            for t in tipos_feat:
                # Se estiver definido como true, guarda as características num arquivo
                if tipos_feat[t]:
                    nome_arq_feat = dir_arq
                
                    # Abre o arquivo com as características
                    arquivo_feat = open(nome_arq_feat, 'a')
                    # Grava cada uma das características no arquivo na pasta de destino, conforme seu tipo
                    for f in feat[t]:
                        print("f: ",f)
                        arquivo_feat.write("%f " % f)
                    # Escreve o nome do arquivo e pula uma linha
                    arquivo_feat.write("\n")
                    # Fecha o arquivo
                    arquivo_feat.close()
    
    print("Terminado, arquivo salvo em {}".format(dir_arq))
