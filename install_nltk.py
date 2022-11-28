import nltk
from nltk.corpus import treebank, brown, conll2000

#nltk.download('treebank')
#nltk.download('brown')
#nltk.download('conll2000')
#nltk.download('universal_tagset')
#nltk.download('tagsets')


treebank_corpus = treebank.tagged_sents(tagset='universal')
brown_corpus = brown.tagged_sents(tagset='universal')
conll_corpus = conll2000.tagged_sents(tagset='universal')
#合併三個資料集
tagged_sentences = treebank_corpus + brown_corpus + conll_corpus

