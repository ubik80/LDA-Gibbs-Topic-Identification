# coding: utf8

# Auswertung der Ergebnisse von gibbsSampler.py und Visualisierung in html

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import pyLDAvis as lvi
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

Tet=np.loadtxt('tetas.txt')
Tet=np.array(Tet)
Tet=normalize(Tet,norm='l2',axis=1)

Phi=np.loadtxt('phis.txt')
Phi=np.array(Phi)

Ttls=[]
file=open("titles.txt","r")
Ts=file.read()
Ttls=Ts.split(";")

Vocs=[]
file=open("vocs.txt","r")
Vocs=file.read()
Vocs=Vocs.split(";")
Vocs=np.array(Vocs)

Docs=[]
file=open("docs.txt","r")
Docs=file.read()
Docs=Docs.split(";")
Docs=np.array(Docs)

################### RECOMMENDER SYSTEM:

i=Ttls.index("Science") #z.B.: Batman, Banana, Science, Angela Merkel

sims=np.zeros(Tet.shape[0],dtype=float)
for j in range(Tet.shape[0]):
    sims[j]=np.dot(Tet[i,:],Tet[j,:])

simsIdx=np.argsort(sims)
sortIdx=np.flip(simsIdx)

print(Ttls[i])

for j in range(0,20):
    print(Ttls[sortIdx[j]])

################### Wörter in Themen:

i=17

topic=Phi[i]
topIdx=np.argsort(topic)
sortIdx=np.flip(topIdx)

for j in range(0,20):
    print(Vocs[sortIdx[j]])

################### pyLDAvis:

docLenght=np.zeros(Docs.shape[0],dtype=int)
for i in range(Docs.shape[0]):
    docLenght[i]=len(Docs[i])

wordFreqDict=dict()
for d in Docs:
    words=d.split()
    for w in words:
        if w in wordFreqDict:
            wordFreqDict[w]+=1
        else:
            wordFreqDict[w]=1
wordFreqDict['']=0

termFreq=np.zeros(Vocs.shape[0],dtype=int)
for i in range(Vocs.shape[0]):
    termFreq[i]=wordFreqDict[Vocs[i]]

data={'topic_term_dists':Phi,'doc_topic_dists':Tet,'doc_lengths':docLenght,'vocab':Vocs,'term_frequency':termFreq}

vis=lvi.prepare(**data)

file = open("vis.html","w")
lvi.save_html(vis,file)
file.close()

################### Histogram über Themen:

h=np.sum(Tet,axis=0)
h=np.array(h)
h=np.divide(h,np.sum(h))

topicNums=[i for i in range(len(h))]
order=np.argsort(h)
order=np.flip(descOrder)

fig, ax = plt.subplots(figsize=(15,5))
ax.bar(range(40),h[order])
ax.set_xticklabels(order)
ax.set_xticks(range(40))
ax.set_ylabel('frequency')
ax.set_xlabel('topics')

fig.savefig('histogram')
