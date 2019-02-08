# coding: utf8

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time as tm
import os
from sklearn.preprocessing import normalize

# Parameter:
K=40 # Anzahl Topics
numOfProc=36

# Dokumente laden:
Docs=[]
file=open("docs.txt","r")
Ds=file.read()
Ds=Ds.split(";")
for D in Ds:
    d=D.split()
    Docs.append(d)
print("documents loaded")

# Vokabular laden:
Vocs=[]
file=open("vocs.txt","r")
Vs=file.read()
Vocs=Vs.split(";")
wordIdx=dict()
for i in range(len(Vocs)):
    wordIdx[Vocs[i]]=i
print("vocabulary loaded")

# Dokumente aus Indizes:
iDocs=[]
for d in Docs:
    id=np.zeros(len(d),dtype=int)
    for i in range(len(d)):
        id[i]=wordIdx[d[i]]
    iDocs.append(id)
iDocs=np.array(iDocs)

def gibsSampling(seed,K,iDocs,Vocs,wordIdx,outputTetas,outputPhis):
    np.random.seed(seed)
    # Parameter:
    M=len(iDocs)
    V=len(Vocs)
    bet=np.ones([V])/V
    betCount=0
    sumBet=np.sum(bet)
    alp=np.ones([K])/K
    alpCount=0
    Tops=[np.zeros(len(d),dtype=int) for d in iDocs]
    Tops=np.asanyarray(Tops)
    # Variablen:
    NMK=np.zeros([M,K],dtype=int)
    NKT=np.zeros([K,V],dtype=int)
    nk=np.zeros([K],dtype=int)
    Tetas=np.zeros([M,K],dtype=float)
    Phis=np.zeros([K,V],dtype=float)
    # Initialisation:
    for m in range(M):
        for n in range(len(iDocs[m])):
            k=np.argwhere(np.random.multinomial(1,alp)==1)[0][0]
            Tops[m][n]=k
            t=iDocs[m][n]
            NMK[m,k]+=1
            NKT[k,t]+=1
            nk[k]+=1
    # Gibbs sampling:
    finished=False
    L=-10
    LL=0 # Anz. gezogene Samples für Mittelwertbildung
    while not finished:
        for m in range(M):
            for n in range(len(iDocs[m])):
                k=Tops[m][n]
                t=iDocs[m][n]
                NMK[m,k]-=1
                NKT[k,t]-=1
                nk[k]-=1
                nenn=nk+sumBet
                pt=NKT[:,t]+bet[t]
                pk=NMK[m,:]+alp
                p=np.multiply(np.divide(pt,nenn),pk)
                p-=np.min(p)
                p/=np.sum(p)
                k_new=np.argwhere(np.random.multinomial(1,p)==1)[0][0]
                Tops[m][n]=k_new
                NMK[m,k_new]+=1
                NKT[k_new,t]+=1
                nk[k_new]+=1
        L+=1
        if L%1==0:
            filename=str(mp.current_process())
            filename=filename[9:19].replace(',','')
            file=open(filename+".txt","w")
            file.write(filename+" LL: "+str(LL)+" L: "+str(L))
            file.close()
        if L>=10:
            Tetas=np.add(Tetas,NMK)
            alpCount+=1
            Phis=np.add(Phis,NKT)
            betCount+=1
            L=0
            LL+=1
            if LL==15:
                finished=True
    # Mittelwertbildung für Teta und Phi:
    alpMat=[alp for mm in range(M)]
    alpMat=np.array(alpMat)*alpCount
    Tetas=np.add(Tetas,alpMat)
    Tetas=normalize(Tetas,axis=1,norm='l1')
    betMat=[bet for kk in range(K)]
    betMat=np.array(betMat)*betCount
    Phis=np.add(Phis,betMat)
    Phis=normalize(Phis,axis=1,norm='l1')
    outputTetas.put(Tetas)
    outputPhis.put(Phis)
    return

# Prozesse anlegen:
outputTetas=mp.Queue()
outputPhis=mp.Queue()
processes=[mp.Process(target=gibsSampling, args=(x*7+43,K,iDocs,Vocs,wordIdx,outputTetas,outputPhis)) for x in range(numOfProc)]

# Sampling Prozesse ausführen:
print("start gibs-sampling")
startTime=tm.time()
for p in processes:
    p.start()
Tetas=[outputTetas.get() for p in processes]
Phis=[outputPhis.get() for p in processes]
for p in processes:
    p.join()
endTime=tm.time()
duration=endTime-startTime
duration
print("gibs-sampling finished, ...duration [min]: "+str(duration/60))

Tetas=np.asanyarray(Tetas)
Phis=np.asanyarray(Phis)
print('results from gibs-sampling collected')

# Permutation, da Themen in unterschiedlicher Reihenfolge vorliegen können:
finalTeta=Tetas[0]
simMat=np.zeros([K,K])
perm=np.zeros([K],dtype=int)
for i in range(1,numOfProc):
    for ii in range(K):
        for iii in range(K):
            sim=np.dot(finalTeta[:,ii],Tetas[i][:,iii])
            sim=sim/(np.linalg.norm(finalTeta[:,ii])*np.linalg.norm(Tetas[i][:,iii]))
            simMat[ii,iii]=sim
    print("Teta simMat finished for Proc. "+str(i))
    for ii in range(K):
        maxIdx=np.unravel_index(np.argmax(simMat),[K,K])
        perm[maxIdx[0]]=maxIdx[1]
        simMat[maxIdx[0],:]=0
        simMat[:,maxIdx[1]]=0
    finalTeta=finalTeta+Tetas[i][:,perm]
    print("permuted Teta from Proc. "+str(i))
print("Teta permutation finished")

# Permutation, da Themen in unterschiedlicher Reihenfolge vorliegen können:
finalPhi=Phis[0]
simMat=np.zeros([K,K])
perm=np.zeros([K],dtype=int)
for i in range(1,numOfProc):
    for ii in range(K):
        for iii in range(K):
            sim=np.dot(finalPhi[ii,:],Phis[i][iii,:])
            sim=sim/(np.linalg.norm(finalPhi[ii,:])*np.linalg.norm(Phis[i][iii,:]))
            simMat[ii,iii]=sim
    print("Phi simMat finished for Proc. "+str(i))
    for ii in range(K):
        maxIdx=np.unravel_index(np.argmax(simMat),[K,K])
        perm[maxIdx[0]]=maxIdx[1]
        simMat[maxIdx[0],:]=0
        simMat[:,maxIdx[1]]=0
    finalPhi=finalPhi+Phis[i][perm,:]
    print("permuted Phi from Proc. "+str(i))
print("Phi permutation finished")

finalTeta=normalize(finalTeta, axis=1, norm='l1')
finalPhi=normalize(finalPhi, axis=1, norm='l1')
np.savetxt("tetas.txt",finalTeta)
np.savetxt("phis.txt",finalPhi)
print("tetas and phis saved to file")
