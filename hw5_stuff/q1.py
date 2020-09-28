import nltk

tagset = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]

print(tagset)