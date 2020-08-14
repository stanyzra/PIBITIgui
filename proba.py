import numpy as np
file_vetTreino = open("vetores/vetTreino.txt","r")
conteudo_vetTreino = file_vetTreino.readlines()
file_vetTreino.close()

# Converte as linhas de string para matriz
linhas = len(conteudo_vetTreino)
vetTreino = [] #vetor que separa as caracteristicas em colunas para facilitar as operações
for i in range(linhas):
    np.array(vetTreino.append(conteudo_vetTreino[i].split()))

file_vet_diss_teste = open("vetores/vetor_diss_teste.txt","r")
conteudo_vetTeste = file_vet_diss_teste.readlines()
file_vet_diss_teste.close()

# Converte as linhas de string para matriz
linhas = len(conteudo_vetTeste)
vetTeste = [] #vetor que separa as caracteristicas em colunas para facilitar as operações
for i in range(linhas):
    np.array(vetTeste.append(conteudo_vetTeste[i].split()))
    
    
from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(vetTreino, vetTeste)

