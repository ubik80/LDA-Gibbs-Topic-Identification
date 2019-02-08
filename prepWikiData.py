# coding: utf8

import xml.etree.ElementTree as et
import gensim.parsing.preprocessing as gs
import re

maxNumToProc=100000000
maxDocLenght_Target=20
minDocLength_Target=10

myStopWords=['file:','wikt:','image:','jpg','|thumb','.gif',
'date=','00px','0px','px','dablink','<ref>','</ref>','citeweb',
'url','http','https','.com','.org','html','htm',
'title=','www','_r=','category:','|right','name =','img','img_capt',
'background =','infobox','dmy','icon.svg','cc','df=yes','(ed)',
'redirect','disambiguation','image_','(converted)','.svg','refn','.gov',
'conventional_long_','0.2em','.au','.cfm','webarchive',
'web.archive','dfat','edition=','pdf','bot:','archive=',
'df=-all','group=','name=','<center>','<br','lower:','.ogg','/center','|map',
'map_width','|_map','type:','location=','page=','hlist','end hlist','small|',
'<ref name','cite web','nowiki','{','}','|','[',']','-','0','1','2','3','4','5',
'6','7','8','9','(',')','=','.',',','?','!',';','_','#','/','&','$','§','"','“',
'nosource','call','png','italic','italictitl']

tree=et.parse('raw.xml')
root=tree.getroot()
pages=root.findall('{http://www.mediawiki.org/xml/export-0.10/}page')
Docs=[]
Ttls=[]
numProc=0
for p in pages:
    tl=p.find('{http://www.mediawiki.org/xml/export-0.10/}title')
    ns=p.find('{http://www.mediawiki.org/xml/export-0.10/}ns')
    if ns.text=='0':
        rev=p.find('{http://www.mediawiki.org/xml/export-0.10/}revision')
        txt=rev.find('{http://www.mediawiki.org/xml/export-0.10/}text')
        text=txt.text
        end=text.find('==')
        text=text[0:end]
        text=str(text).lower()
        Docs.append(text)
        Ttls.append(tl.text)
        numProc=numProc+1
        if numProc==maxNumToProc:
            break
print('xml parsed')

myStopWords=list(set(myStopWords))
myStopWords.sort(key=lambda s: -len(s))
for i in range(len(Docs)):
    for sw in myStopWords:
        Docs[i]=Docs[i].replace(sw,' ')
print('own stopwords removed')

Docs=gs.preprocess_documents(Docs)
print('gs-preprocessing done')

wordFreq=dict()
for d in Docs:
    for w in d:
        if w in wordFreq:
            wordFreq[w]+=1
        else:
            wordFreq[w]=1

totNumOfWord=0
for i in range(len(Docs)):
    totNumOfWord=totNumOfWord+len(Docs[i])

lowLim=totNumOfWord*0.00001
highLim=totNumOfWord*0.002
i=0
while i < len(Docs):
    ii=0
    while ii < len(Docs[i]):
        freq=wordFreq[Docs[i][ii]]
        if freq<lowLim or freq>highLim:
            del Docs[i][ii]
        else:
            ii=ii+1
    i=i+1
print('eliminated words by frequency')

i=0
while i < len(Docs):
    if len(Docs[i])<minDocLength_Target:
        del Docs[i]
        del Ttls[i]
    else:
        i=i+1
print('small documents removed')

for i in range(len(Docs)):
    endIdx=min(len(Docs[i]),maxDocLenght_Target)
    Docs[i]=Docs[i][0:endIdx]
print('document length limited')

print('build up vocabulary:')
Vocs=set()
for i in range(len(Docs)):
    Vocs=Vocs|set(Docs[i])
    if i%10000==0:
        print(i)
print('vocabulary extracted')

Vocs=list(Vocs)
for i in range(len(Vocs)):
    Vocs[i]=re.sub(r'[^\x00-\x7f]','',Vocs[i])
Vocs=list(set(Vocs))
if '' in Vocs:
    Vocs.remove('')
print('vocabulary cleaned (re)')

file = open("docs.txt","w")
for d in Docs:
    text=' '.join(d)
    text=re.sub(r'[^\x00-\x7f]','',text)
    text=text+";"
    file.write(text)
file.close()
print('saved documents to file')

file = open("titles.txt","w")
for t in Ttls:
    text=re.sub(r'[^\x00-\x7f]','',t)
    text=text+";"
    file.write(text)
file.close()
print('saved titles to file')

file = open("vocs.txt","w")
for v in Vocs:
    file.write(v+";")
file.close()
print('saved vocabulary to file')
