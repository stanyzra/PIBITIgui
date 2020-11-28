from pathlib import Path
import os, sys, re, shutil
import numpy as np
from rp_extract import rp_extract
#import extracao_audio
import numpy as np
from scipy.io.wavfile import read
import librosa

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def organizarDados(path, destino, audios):
    print("Organizando dados")
    # Diretórios usados para consulta da base e criação dos folds
    dir_origem = path
    dir_gravacoes = dir_origem
    dir_origem += destino
    dir_gravacoes += 'audios_gravados'
    cont_audios = audios
    
    # Onde serão armazenados os arquivos gerados (mesma estrutura de divisão)
    numPessoas = 5
    # Lista os arquivos da pasta e ordena
    conteudo = os.listdir(dir_origem)
    conteudo = natural_sort(conteudo)
    
    nome_audio = os.listdir(dir_gravacoes)
    nome_audio = natural_sort(nome_audio)
    
    
    sys.path.append('./rp_extract')
    tipos_feat = {'rh' : False, 'rp' : False, 'ssd': True}
    #dir_treino = dir_destino+'treino-'+bases[decisao]+'.txt'
    dir_treino = "arquivos_treino/"
    bases_treino = os.listdir(dir_treino)
    dir_treino += bases_treino[0]+'/treino-{}.txt'.format(destino)
    
    print("Extraindo dados do áudio: ")
    for i in range(5):
        print('{}.wav'.format(cont_audios+1))
        dir_extracao = dir_gravacoes+'/{}.wav'.format(cont_audios+1)
        wavedata, samplerate = librosa.load(dir_extracao, sr=44100)
        feat = rp_extract.rp_extract(wavedata, samplerate, extract_rp=tipos_feat['rp'], extract_ssd=tipos_feat['ssd'], extract_rh=tipos_feat['rh'])
        cont_audios += 1
     # Verifica cada uma dos três tipos de features acústucas
        for t in tipos_feat:
            # Se estiver definido como true, guarda as características num arquivo
            if tipos_feat[t]:
                nome_arq_feat = dir_treino
            
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
    
    atual = 0
    #print(audios)
    for pastas in range(int(audios/5), int(len(conteudo)+1)):
    # Não esquece de adicionar a barra pra direita no final do caminho
        cont = 0
        print(pastas)
        
        dir_destino = dir_origem+'/' + 's' + str((pastas)+1) + '/'
    
        if not os.path.exists(dir_destino): #Cria as pastas s1, s2, ... , sn caso não existam
            os.makedirs(dir_destino)
    
         
        for i in range(5):
            
            dir_pessoas = dir_gravacoes+'/' + nome_audio[atual]
            atual += 1
             
            if cont < numPessoas:
                 shutil.move(dir_pessoas, dir_destino)
                 cont += 1
                
    
#organizarDados('PIBITI/', 'brSD_audiofeat_audios', 425)
"""

print("Organizando dados")
# Diretórios usados para consulta da base e criação dos folds
destino = 'brSD_audiofeat_audios'
dir_origem = 'PIBITI/'
dir_gravacoes = dir_origem
dir_origem += destino
dir_gravacoes += 'audios_gravados/'
audios = 425
cont_audios = audios
# Onde serão armazenados os arquivos gerados (mesma estrutura de divisão)
numPessoas = 5
# Lista os arquivos da pasta e ordena
conteudo = os.listdir(dir_origem)
conteudo = natural_sort(conteudo)

nome_audio = os.listdir(dir_gravacoes)
nome_audio = natural_sort(nome_audio)

sys.path.append('./rp_extract')
tipos_feat = {'rh' : False, 'rp' : False, 'ssd': True}
#dir_treino = dir_destino+'treino-'+bases[decisao]+'.txt'
dir_treino = "arquivos_treino/"
bases_treino = os.listdir(dir_treino)
dir_treino += bases_treino[0]+'/treino-{}.txt'.format(destino)

print("Extraindo dados do áudio: ")
for i in range(5):
    print('{}.wav'.format(cont_audios+1))
    dir_extracao = dir_gravacoes+'{}.wav'.format(cont_audios+1)
    wavedata, samplerate = librosa.load(dir_extracao, sr=44100)
    feat = rp_extract.rp_extract(wavedata, samplerate, extract_rp=tipos_feat['rp'], extract_ssd=tipos_feat['ssd'], extract_rh=tipos_feat['rh'])
    cont_audios += 1
 # Verifica cada uma dos três tipos de features acústucas
    for t in tipos_feat:
        # Se estiver definido como true, guarda as características num arquivo
        if tipos_feat[t]:
            nome_arq_feat = dir_treino
        
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

atual = 0
#print(audios)
for pastas in range(int(audios/5), int(len(conteudo)+1)):
# Não esquece de adicionar a barra pra direita no final do caminho
    cont = 0
    print(pastas)
    
    dir_destino = dir_origem+'/' + 's' + str((pastas)+1) + '/'

    if not os.path.exists(dir_destino): #Cria as pastas s1, s2, ... , sn caso não existam
        os.makedirs(dir_destino)

     
    for i in range(5):
        
        dir_pessoas = dir_gravacoes + nome_audio[atual]
        atual += 1
         
        if cont < numPessoas:
             shutil.move(dir_pessoas, dir_destino)
             cont += 1
"""