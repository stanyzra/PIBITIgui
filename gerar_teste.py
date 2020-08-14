import numpy as np

def gerar(vetor_teste):
    
    x = np.array(vetor_teste)
    vetor_diss = []
    linha = len(x)
    #print(range(linha))
    #print(range(len(linha)))
    
    for i in range(linha):
        for j in range(linha):
#            if i == j:
#                continue
            v = abs(np.float16(x[i]) - np.float16(x[j]))
            vetor_diss.append(v)
            
   #print(vetor_diss)
       
    file_vetor_diss = open("vetor_diss_teste.txt","w")
   
    conteudo_diss = vetor_diss
    
    linhas = len(conteudo_diss)
    colunas = len(conteudo_diss[0])

    num_amostras_teste = 160

    for i in range(linhas):


        for j in range(colunas):

            file_vetor_diss.write("{} ".format(conteudo_diss[i][j]))
            
        if i < num_amostras_teste*num_amostras_teste:
            file_vetor_diss.write("\n")

    file_vetor_diss.close()
               