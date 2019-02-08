# LDA-Gibbs-Topic-Identification
Identification of topics in the simple english wikipdia using Gibbs sampling 

- LDAGibbsTopicIdentification.pdf:  project report <- start reading this to understand the project
- gibbsSampler.py:  this is the core of the project - the gibbs sampler
- prepWikiData.py:  preprocessing of the simple wikipedia data (from .xml file), run this to create all input files necessary
                    for gibbsSampler.py
- evaluation.py: play with the results from gibbsSampler.py
- pyLDAvis.html: visualisation
- generateDocs.py: create some documents with LDA model (for testing of gibbsSampler.py)
