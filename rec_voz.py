from sys import byteorder
from array import array
from struct import pack
import librosa, re, pyaudio, wave, organizar_dados, os, sys, copy, shutil
from rp_extract import rp_extract
#import extracao_audio
import numpy as np
from scipy.io.wavfile import read
import librosa, shutil
import math
import converte_e_classifica
os.chdir(r'../pibiti/')
sys.stdout.reconfigure(encoding='utf-8')

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

THRESHOLD = 500  # audio levels not normalised.
CHUNK_SIZE = 1024
SILENT_CHUNKS = 3 * 44100 / 1024  # about 3sec
FORMAT = pyaudio.paInt16
FRAME_MAX_VALUE = 2 ** 15 - 1
NORMALIZE_MINUS_ONE_dB = 10 ** (-1.0 / 20)
RATE = 44100
CHANNELS = 1
TRIM_APPEND = RATE / 4

def is_silent(data_chunk):
    """Returns 'True' if below the 'silent' threshold"""
    return max(data_chunk) < THRESHOLD

def normalize(data_all):
    """Amplify the volume out to max -1dB"""
    # MAXIMUM = 16384
    normalize_factor = (float(NORMALIZE_MINUS_ONE_dB * FRAME_MAX_VALUE)
                        / max(abs(i) for i in data_all))

    r = array('h')
    for i in data_all:
        r.append(int(i * normalize_factor))
    return r

def trim(data_all):
    _from = 0
    _to = len(data_all) - 1
    for i, b in enumerate(data_all):
        if abs(b) > THRESHOLD:
            _from = max(0, i - TRIM_APPEND)
            break

    for i, b in enumerate(reversed(data_all)):
        if abs(b) > THRESHOLD:
            _to = min(len(data_all) - 1, len(data_all) - 1 - i + TRIM_APPEND)
            break

    return copy.deepcopy(data_all[int(_from):(int(_to) + 1)])


def record():
    """Record a word or words from the microphone and 
    return the data as an array of signed shorts."""

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK_SIZE)

    silent_chunks = 0
    audio_started = False
    data_all = array('h')

    while True:
        # little endian, signed short
        data_chunk = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            data_chunk.byteswap()
        data_all.extend(data_chunk)

        silent = is_silent(data_chunk)

        if audio_started:
            if silent:
                silent_chunks += 1
                if silent_chunks > SILENT_CHUNKS:
                    break
            else: 
                silent_chunks = 0
        elif not silent:
            audio_started = True              

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    data_all = trim(data_all)  # we trim before normalize as threshhold applies to un-normalized wave (as well as is_silent() function)
    data_all = normalize(data_all)
    return sample_width, data_all

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h' * len(data)), *data)

    wave_file = wave.open(path, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(sample_width)
    wave_file.setframerate(RATE)
    wave_file.writeframes(data)
    wave_file.close()
    
def extrair_e_converter():
    
    sys.path.append('./rp_extract')
    tipos_feat = {'rh' : False, 'rp' : False, 'ssd': True}
    print("Extraindo dados do áudio gravado")
    wavedata, samplerate = librosa.load(dir_entrada_teste, sr=44100)
    feat = rp_extract.rp_extract(wavedata, samplerate, extract_rp=tipos_feat['rp'], extract_ssd=tipos_feat['ssd'], extract_rh=tipos_feat['rh'])
                  # Verifica cada uma dos três tipos de features acústucas
    for t in tipos_feat:
      # Se estiver definido como true, guarda as características num arquivo
      if tipos_feat[t]:
          nome_arq_feat = dir_arq+'.txt'
      
          # Abre o arquivo com as características
          arquivo_feat = open(nome_arq_feat, 'w')
          # Grava cada uma das características no arquivo na pasta de destino, conforme seu tipo
          for f in feat[t]:
              print("f: ",f)
              arquivo_feat.write("%f " % f)
          # Escreve o nome do arquivo e pula uma linha
          arquivo_feat.write("\n")
          # Fecha o arquivo
          arquivo_feat.close()
          print("convertendo para entrada libSVM")
        
    dir_conversao = r"arquivos_teste"
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
    treino = dir_conversao+r'\convertido-audio_extraido.svm'
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

def encontrar_pessoa():

    dir_arq_teste = r"arquivos_teste"
    if not os.path.exists(dir_arq_teste):
        print("Criando pasta de arquivos para teste")
        os.mkdir(dir_arq_teste)
        
    # Indo para a pasta do libsvm
    dir_svm = r".\libsvm-3.24\windows"
    os.chdir(dir_svm)
    
    extensoes = ['model','range','scale','scale2','predict','svm']
    #nome_txt = [_ for _ in os.listdir(dir_conversao) if _.endswith(extensao_txt)]
    
    dir_treino = r"..\..\resultados"
    resultados = [_ for _ in os.listdir(dir_treino) if _.endswith(extensoes[1])]
    dir_treino += r"\{}".format(resultados[0])
    
    dir_teste = r"..\..\{}".format(dir_arq_teste)
    teste = [_ for _ in os.listdir(dir_teste) if _.endswith(extensoes[5])]
    dir_teste += r"\{}".format(teste[0])
    
    comando_scale = ".\svm-scale -r {} {} > {}.{}".format(dir_treino, dir_teste, dir_teste,extensoes[3])
    #print(comando_scale)
    os.system(comando_scale)
    
    dir_destino = r"..\..\resultados"
    comando_copy = "move /Y {}.{} {}".format(dir_teste, extensoes[3], dir_destino)
    os.system(comando_copy)
    #print(comando_copy)
    
    ########################################################################################################
    dir_modelo = r"..\..\resultados"
    modelo = [_ for _ in os.listdir(dir_modelo) if _.endswith(extensoes[0])]
    dir_modelo += r"\{}".format(modelo[0])
    
    dir_scale2 = r"..\..\resultados"
    scale2 = [_ for _ in os.listdir(dir_scale2) if _.endswith(extensoes[3])]
    dir_scale2 += r"\{}".format(scale2[0])
    
    dir_teste = r"..\..\{}".format(dir_arq_teste)
    teste = [_ for _ in os.listdir(dir_teste) if _.endswith(extensoes[5])]
    dir_teste += r"\{}".format(teste[0])
    
    comando_predict = ".\svm-predict -b 1 {} {} {}.{}".format(dir_scale2, dir_modelo, dir_teste, extensoes[4])
    os.system(comando_predict)
    #print(comando_predict)
    
    comando_copy = "move /Y {}.{} {}".format(dir_teste, extensoes[4], dir_destino)
    os.system(comando_copy)
    #print(comando_copy)
    
    
    dir_arq_predict = r"..\..\resultados"
    arq_predict = [_ for _ in os.listdir(dir_arq_predict) if _.endswith(extensoes[4])]
    dir_arq_predict += r"\{}".format(arq_predict[0]) 
    
    file_predict = open(dir_arq_predict, 'r')
    
    conteudo = file_predict.readlines()
    
    file_predict.close()

    linha = len(conteudo)
    x = []
    for i in range(1, linha):
        np.array(x.append(conteudo[i].split()))
    
    x[0][0] = 0
    
    #label = np.argmax(x)
    suspeitos = []

    for cont in range(3):
        label = (np.argmax(x))
        suspeitos.append(label)
        x[0][label] = 0
    os.chdir(r"..\..\..\pibiti")
    print("action && result && O sistema acredita que possa ser uma das três pessoas a seguir: (TOP 3) {}".format(suspeitos))
    sys.stdout.flush()

def pegar_audios(x):
    x = int(x)
    cont = 1
    dir_pessoas = r"arquivos_treino"
    base = os.listdir(dir_pessoas)
    #print("base: ", base)
    dir_pessoas += r"\{}".format(base[0])
    extensao_txt = ".txt"
    nome_txt = [_ for _ in os.listdir(dir_pessoas) if _.endswith(extensao_txt)]
     
    #print("DIR PESSOAS: ", dir_pessoas)
    file_pessoas = open(r"{}".format(dir_pessoas) + r"\{}".format(nome_txt[0]), 'r')
    print(r"{}".format(dir_pessoas) + r"\{}".format(nome_txt[0]))
    conteudo = file_pessoas.readlines()
    conteudo = natural_sort(conteudo)        
    file_pessoas.close()    
    
    audios = len(conteudo)
    
    if(x == 1):
        for i in range(5):
            audios += 1
    elif(x == 2):
        while(True):
            print("Por favor, leia este trecho: ")
            print("action && text && {}".format(cont-1))
            sys.stdout.flush()
            record_to_file((r"PIBITI\audios_gravados\{}".format(audios+1))+'.wav')
            cont += 1
            audios += 1
            if(cont == 6): 
                print("action && text_without && Gravação finalizada")
                sys.stdout.flush()
                print("action && loading && true && Realizando classificação")
                sys.stdout.flush()
                print("Terminado")
                break
    return audios

if __name__ == '__main__':
    cont = 1
    classe = 1
    dir_origem = 'PIBITI/'
    if not os.path.exists(dir_origem +'audios_gravados/'):
        print("Criando pasta para guardar as gravações")
        os.mkdir(dir_origem+'audios_gravados/')
        print("Terminado")
    while(True):
        print("1 - incluir pessoa por upload\n2 - incluir pessoa por gravação\n3 - gravar uma amostra para teste\n4 - upar uma amostra já gravada para teste")
        comando = (input())
        if(comando == ""):
            print("Saindo do programa")
            sys.exit()
        elif(int(comando) == 4):
            dir_entrada_teste = "audios_upados_teste/entrada_upada_teste.wav"
            if not os.path.exists("audios_upados_teste/"):
                print("Criando pasta para guardar audio upado")
                os.mkdir("audios_upados_teste/")
                
            dir_arq = "arquivos_teste/audio_extraido"
            extrair_e_converter()
            encontrar_pessoa()
    
        elif(int(comando) == 3):
            print("Por favor, diga a sua frase (6 segundos, no minimo)")
            dir_entrada_teste = "audios_gravados_teste/entrada_teste.wav"
            if not os.path.exists("audios_gravados_teste/"):
                print("Criando pasta para guardar audio gravado")
                os.mkdir("audios_gravados_teste/")
                
            dir_arq = "arquivos_teste/audio_extraido"
            record_to_file(dir_entrada_teste)
            extrair_e_converter()
            encontrar_pessoa()
        elif(int(comando) == 2):
            
            audios = pegar_audios(comando)     
            #print(audios)
            
            organizar_dados.organizarDados('PIBITI/', 'brSD_audiofeat_audios', audios-5)
            print("Convertendo e classificando")
            converte_e_classifica.converterEClassificar(r"arquivos_treino", r"brSD_audiofeat_audios")
            print("action && result && Pessoa incluida com sucesso")
            sys.stdout.flush()
        elif(int(comando) == 1):
            #IMPORTAR ARQUIVO COM INTERFACE, POIS NO TERMINAL É SÓ ARRASTAR OS ARQUIVOS
            audios = pegar_audios(comando)
            #print(audios)

            organizar_dados.organizarDados('PIBITI/', 'brSD_audiofeat_audios', audios-5)
            print("Convertendo e classificando")
            converte_e_classifica.converterEClassificar(r"arquivos_treino", r"brSD_audiofeat_audios")

