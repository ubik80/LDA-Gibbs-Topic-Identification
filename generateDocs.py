# Generierung von synthetischen Dokumenten für definierte Testbedingungen

import numpy as np
import matplotlib.pyplot as plt
import pickle
import re

def genVocabFromTXTFile(filename):
    d =[]
    with open(filename) as f:
        for line in f:
            line=line.lower()
            line=re.sub(r'\W+', ' ', line)
            l = line.split()
        for word in l:
            d.append(word)
    d=list(set(l))
    return d

def generator(M,K,vocab,alpha,beta,nue):
    tetas=np.random.dirichlet(alpha,M) # Verteilung über Topics für jedes Dok.
    phis=np.random.dirichlet(beta,K) # Verteilung über Wörter für jedes Topic
    Ns=np.random.poisson(nue,size=M) # Länge der Dokumente
    # Für Rückgabe:
    Docs=[]
    Topics=[]
    for i in range(M): # Für jedes Dokument
        # Für jedes Wort im Dokument ein Toppic:
        wordTopics=np.argwhere(np.random.multinomial(1, tetas[i], Ns[i])==1)[:,1] # Vektor mit Topic-Nr. pro Wort
        w=[]
        t=[]
        for ii in range(Ns[i]): # Für jedes Wort im Dokument
            # Für jedes Wort im Dokument ein Wort aus dem Vokabular:
            wordIdx=np.argwhere(np.random.multinomial(1, phis[wordTopics[ii]])==1)[0][0]
            word=vocab[wordIdx]
            w.append(word)
            t.append(wordTopics[ii])
        Docs.append(w)
        Topics.append(t)
    return [Docs, Topics, tetas, phis];

vocab=genVocabFromTXTFile('randomText.txt')
V=len(vocab)
M=30 # number of documents
K=3 # number of topics
alp=np.ones(K)/K # Verteilung Topics für jedes Dokument
bet=np.ones(V)/V # Verteilung Wörter für jedes Topic
nu=5 # Für die Verteilung der Dokumenten Länge

[Docs, Topics, tetas, phis]=generator(M,K,vocab,alp,bet,nu)

file = open("docs.txt","w")
for d in Docs:
    text=""
    for w in d:
        text=text+w+" "
    text=text+";"
    file.write(text)
file.close()

file = open("vocs.txt","w")
for v in vocab:
    file.write(v+";")
file.close()
print('saved vocabulary to file')
