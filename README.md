# LDA-Gibbs-Topic-Identification
Identification of topics in the simple english wikipdia using Gibbs sampling 

- LDAGibbsTopicIdentification.pdf:  project report <- Start reading this to understand the project.
- prepWikiData.py:  preprocessing of the simple english wikipedia data (input: raw.xml). Run this to create all input 
                    files necessary for gibbsSampler.py. Download simplewiki-<date>-pages-meta-current.xml.bz2 from simple 
                    english wikipedia, unpack and rename to raw.xml as input.
- gibbsSampler.py:  This is the core of the project - the gibbs sampler.
- evaluation.py:  Play with the results from gibbsSampler.py.
- pyLDAvis.html:  visualisation
- generateDocs.py:  Create some documents with LDA model (for testing of gibbsSampler.py).
