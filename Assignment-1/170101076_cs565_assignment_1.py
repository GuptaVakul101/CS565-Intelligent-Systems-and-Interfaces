# -*- coding: utf-8 -*-
"""170101076-CS565-Assignment-1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1b67oKwJhbwfGoFDw7wlUfJxXfyuOGrKi

#### **VAKUL GUPTA - 170101076**
# **INTELLIGENT SYSTEMS AND INTERFACES - CS565 - ASSIGNMENT 1**
"""

from google.colab import drive
drive.mount("/content/drive")

"""### 1.3.1 - Analysis using existing NLP tools

#### 0 - PREREQUISITE

##### Install Basic Libraries
"""

!pip install indic-nlp-library
from nltk.probability import FreqDist
import nltk
import codecs
import numpy as np
from nltk.util import ngrams
from matplotlib import pyplot
import math
nltk.download('punkt')
!pip install stanza
import stanza
stanza.download('hi')
from indicnlp.tokenize import sentence_tokenize
from nltk.tokenize import TreebankWordTokenizer, WordPunctTokenizer
from indicnlp.tokenize import indic_tokenize

"""##### Read Input Files"""

englishText = codecs.open('/content/drive/My Drive/Intelligent_Systems_CS565/Assignment-1/en_wiki.txt', 'r').read()
hindiText = codecs.open('/content/drive/My Drive/Intelligent_Systems_CS565/Assignment-1/hi_wiki.txt', 'r').read()
filterRatio = 0.05
englishFilteredText = englishText[0:int(filterRatio*len(englishText))]
hindiFilteredText = hindiText[0:int(filterRatio*len(hindiText))]
print((len(englishFilteredText)/len(englishText))*100)
print((len(hindiFilteredText)/len(hindiText))*100)

"""#### 1 - Sentence Segmentation and Word Tokenization

##### Sentence Segmentation - English

###### First Method
"""

englishSentTokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
englishSentTokens = englishSentTokenizer.tokenize(englishText)
print(len(englishSentTokens))
print(englishSentTokens[0:10])

"""###### Second Method"""

englishSentTokens = nltk.tokenize.regexp_tokenize(englishText, pattern='(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', gaps=True)
print(len(englishSentTokens))
print(englishSentTokens[0:10])

"""##### Sentence Segmentation - Hindi

###### First Method
"""

hindiSentTokens = sentence_tokenize.sentence_split(hindiText, lang='hi')
print(len(hindiSentTokens))
print(hindiSentTokens[0:10])

"""###### Second Method"""

hindiSentTokenizer = stanza.Pipeline(lang='hi', processors='tokenize')
hindiSentTokens = hindiSentTokenizer(hindiFilteredText).sentences
hindiSentTokens = [sentence.text for sentence in hindiSentTokens]
print(len(hindiSentTokens))
print(hindiSentTokens[0:10])

"""##### Word Tokenization - English

###### First Method
"""

englishWordTokenizer = TreebankWordTokenizer()
englishWordTokens = englishWordTokenizer.tokenize(englishText)
print(len(englishWordTokens))
print(englishWordTokens[0:100])

englishWordTokenizer = WordPunctTokenizer() 
englishWordTokens = englishWordTokenizer.tokenize(englishText)
print(len(englishWordTokens))
print(englishWordTokens[0:100])

"""###### Second Method

##### Word Tokenization - Hindi

###### First Method
"""

hindiWordTokenizer = stanza.Pipeline(lang='hi', processors='tokenize', tokenize_no_ssplit=True)
hindiSentTokens = hindiWordTokenizer(hindiFilteredText).sentences
hindiWordTokens = []
for i, sentence in enumerate(hindiSentTokens):
  for token in sentence.tokens:
    hindiWordTokens.append(token.text)
print(len(hindiWordTokens))
print(hindiWordTokens[0:10])

"""###### Second Method"""

hindiWordTokens = indic_tokenize.trivial_tokenize(hindiText)
print(len(hindiWordTokens))
print(hindiWordTokens[0:100])

"""#### Utility Functions [ Analysing N-Grams and Plotting Frequency Distribution ]"""

def extractNGrams(data, num):
  n_grams = ngrams(data, num)
  return [' '.join(grams) for grams in n_grams]

def plotFreqDist(freq_dist, n, lang="english", num_samples=100):
    frequencies = [freq_dist[sample] for sample,_ in freq_dist.most_common(num_samples)]
    pos = np.arange(num_samples)
    width = 1.0
    
    ngramLabel = ""
    if n == 1:
      ngramLabel = lang + "Unigram"
    elif n == 2:
      ngramLabel = lang + "Bigram"
    elif n == 3:
      ngramLabel = lang + "Trigram"

    ax = pyplot.axes()
    ax.set_xticks(pos + (width / 2))
    ax.set_xticklabels(pos)
    ax.set_xlabel(ngramLabel + " Rank")
    ax.set_ylabel(ngramLabel + " Frequency")
    ax.set_title('Frequency Plot of ' + ngramLabel + ' VS Rank')
    ax.grid(True)
    pyplot.ylim(0, math.ceil(max(frequencies)/5)*5)
    pyplot.bar(pos, frequencies, width, color='r', edgecolor='k')
    figure = pyplot.gcf()
    figure.savefig(ngramLabel, dpi=figure.dpi)
    pyplot.close()

"""#### 2 - Unigrams

##### Unigrams - English
"""

englishUniGrams = extractNGrams(englishWordTokens, 1)
print(len(englishUniGrams))
print(englishUniGrams[0:100])

englishUniFreqDist = nltk.FreqDist(englishUniGrams)
plotFreqDist(englishUniFreqDist, 1)
print('Unique Count: ', len(englishUniFreqDist))
print('5 Most Frequent occurring: ', englishUniFreqDist.most_common(5))

"""##### Unigrams - Hindi"""

hindiUnigrams = extractNGrams(hindiWordTokens, 1)
print(len(hindiUnigrams))
print(hindiUnigrams[0:100])

hindiUniFreqDist = nltk.FreqDist(hindiUnigrams)
# plotFreqDist(hindiUniFreqDist, 1, "hindi")
print('Unique Count: ', len(hindiUniFreqDist))
print('5 Most Frequent occurring: ', hindiUniFreqDist.most_common(5))

"""#### 3 - Bigrams

##### Bigrams - English
"""

englishBiGrams = extractNGrams(englishWordTokens, 2)
print(len(englishBiGrams))
print(englishBiGrams[0:100])

englishBiFreqDist = nltk.FreqDist(englishBiGrams)
# plotFreqDist(englishBiFreqDist, 2)
print('Unique Count: ', len(englishBiFreqDist))
print('5 Most Frequent occurring: ', englishBiFreqDist.most_common(5))

"""##### Bigrams - Hindi"""

hindiBigrams = extractNGrams(hindiWordTokens, 2)
print(len(hindiBigrams))
print(hindiBigrams[0:100])

hindiBiFreqDist = nltk.FreqDist(hindiBigrams)
# plotFreqDist(hindiBiFreqDist, 2, "hindi")
print('Unique Count: ', len(hindiBiFreqDist))
print('5 Most Frequent occurring: ', hindiBiFreqDist.most_common(5))

"""#### 4 - Trigrams

##### Trigrams - English
"""

englishTrigrams = extractNGrams(englishWordTokens, 3)
print(len(englishTrigrams))
print(englishTrigrams[0:100])

englishTriFreqDist = nltk.FreqDist(englishTrigrams)
# plotFreqDist(englishTriFreqDist, 3)
print('Unique Count: ', len(englishTriFreqDist))
print('5 Most Frequent occurring: ', englishTriFreqDist.most_common(5))

"""##### Trigrams - Hindi"""

hindiTrigrams = extractNGrams(hindiWordTokens, 3)
print(len(hindiTrigrams))
print(hindiTrigrams[0:100])

hindiTriFreqDist = nltk.FreqDist(hindiTrigrams)
# plotFreqDist(hindiTriFreqDist, 3, "hindi")
print('Unique Count: ', len(hindiTriFreqDist))
print('5 Most Frequent occurring: ', hindiTriFreqDist.most_common(5))

"""### 1.3.2 - Few Basic Questions

#### DATA ANALYSIS WITHOUT STEMMING

#### 1 - Unigrams required for 90% coverage

##### English
"""

englishUniFreqList = np.array(list(reversed(sorted([val for _, val in englishUniFreqDist.items()]))))
uniThresholdCount = np.argmin(englishUniFreqList.cumsum() < englishUniFreqDist.N()*0.9)
print('Number of Unigrams required for 90% coverage: ' + str(uniThresholdCount))

"""##### Hindi"""

hindiUniFreqList = np.array(list(reversed(sorted([val for _, val in hindiUniFreqDist.items()]))))
uniThresholdCount = np.argmin(hindiUniFreqList.cumsum() < hindiUniFreqDist.N()*0.9)
print('Number of Unigrams required for 90% coverage: ' + str(uniThresholdCount))

"""#### 2 - Bigrams required for 80% coverage

##### English
"""

englishBiFreqList = np.array(list(reversed(sorted([val for _, val in englishBiFreqDist.items()]))))
biThresholdCount = np.argmin(englishBiFreqList.cumsum() < englishBiFreqDist.N()*0.8)
print('Number of Bigrams required for 80% coverage: ' + str(biThresholdCount))

"""##### Hindi"""

hindiBiFreqList = np.array(list(reversed(sorted([val for _, val in hindiBiFreqDist.items()]))))
biThresholdCount = np.argmin(hindiBiFreqList.cumsum() < hindiBiFreqDist.N()*0.8)
print('Number of Bigrams required for 80% coverage: ' + str(biThresholdCount))

"""#### 3 - Trigrams required for 70% coverage

##### English
"""

englishTriFreqList = np.array(list(reversed(sorted([val for _, val in englishTriFreqDist.items()]))))
triThresholdCount = np.argmin(englishTriFreqList.cumsum() < englishTriFreqDist.N()*0.7)
print('Number of Trigrams required for 70% coverage: ' + str(triThresholdCount))

"""##### Hindi"""

hindiTriFreqList = np.array(list(reversed(sorted([val for _, val in hindiTriFreqDist.items()]))))
triThresholdCount = np.argmin(hindiTriFreqList.cumsum() < hindiTriFreqDist.N()*0.7)
print('Number of Trigrams required for 70% coverage: ' + str(triThresholdCount))

"""#### DATA ANALYSIS WITH STEMMING"""

from nltk.stem import PorterStemmer
ps = PorterStemmer()

englishWordTokensStemmed = [ps.stem(word) for word in englishWordTokens]

def generate_hin_stem_words(word):
  suffixes = {
    1: [u"ो",u"े",u"ू",u"ु",u"ी",u"ि",u"ा"],
    2: [u"कर",u"ाओ",u"िए",u"ाई",u"ाए",u"ने",u"नी",u"ना",u"ते",u"ीं",u"ती",u"ता",u"ाँ",u"ां",u"ों",u"ें"],
    3: [u"ाकर",u"ाइए",u"ाईं",u"ाया",u"ेगी",u"ेगा",u"ोगी",u"ोगे",u"ाने",u"ाना",u"ाते",u"ाती",u"ाता",u"तीं",u"ाओं",u"ाएं",u"ुओं",u"ुएं",u"ुआं"],
    4: [u"ाएगी",u"ाएगा",u"ाओगी",u"ाओगे",u"एंगी",u"ेंगी",u"एंगे",u"ेंगे",u"ूंगी",u"ूंगा",u"ातीं",u"नाओं",u"नाएं",u"ताओं",u"ताएं",u"ियाँ",u"ियों",u"ियां"],
    5: [u"ाएंगी",u"ाएंगे",u"ाऊंगी",u"ाऊंगा",u"ाइयाँ",u"ाइयों",u"ाइयां"],
  }
  
  for L in 5, 4, 3, 2, 1:
    if len(word) > L + 1:
      for suf in suffixes[L]:
        if word.endswith(suf):
          return word[:-L]
  return word

print('Number of word tokens as input:', len(hindiWordTokens))
hindiWordTokensStemmed = [generate_hin_stem_words(word) for word in hindiWordTokens]
print(len(hindiWordTokensStemmed))
print('Sample Output:', hindiWordTokensStemmed[0:30])

"""##### Unigrams required for 90% coverage - English"""

print(len(englishWordTokensStemmed))
englishUniGramsStemmed = extractNGrams(englishWordTokensStemmed, 1)
print(len(englishUniGramsStemmed))
englishUniFreqDistStemmed = nltk.FreqDist(englishUniGramsStemmed)
englishUniFreqListStemmed = np.array(list(reversed(sorted([val for _, val in englishUniFreqDistStemmed.items()]))))
uniThresholdCountStemmed = np.argmin(englishUniFreqListStemmed.cumsum() < englishUniFreqDistStemmed.N()*0.9)
print('Number of Unigrams required for 90% coverage after stemming: ' + str(uniThresholdCountStemmed))
print('Unique Count: ', len(englishUniFreqDistStemmed))
print('5 Most Frequent occurring: ', englishUniFreqDistStemmed.most_common(5))

"""##### Unigrams required for 90% coverage - Hindi"""

print(len(hindiWordTokensStemmed))
hindiUniGramsStemmed = extractNGrams(hindiWordTokensStemmed, 1)
print(len(hindiUniGramsStemmed))
hindiUniFreqDistStemmed = nltk.FreqDist(hindiUniGramsStemmed)
hindiUniFreqListStemmed = np.array(list(reversed(sorted([val for _, val in hindiUniFreqDistStemmed.items()]))))
uniThresholdCountStemmed = np.argmin(hindiUniFreqListStemmed.cumsum() < hindiUniFreqDistStemmed.N()*0.9)
print('Number of Unigrams required for 90% coverage after stemming: ' + str(uniThresholdCountStemmed))
print('Unique Count: ', len(hindiUniFreqDistStemmed))
print('5 Most Frequent occurring: ', hindiUniFreqDistStemmed.most_common(5))

"""##### Bigrams required for 80% coverage - English"""

englishBiGramsStemmed = extractNGrams(englishWordTokensStemmed, 2)
print(len(englishBiGramsStemmed))
englishBiFreqDistStemmed = nltk.FreqDist(englishBiGramsStemmed)
englishBiFreqListStemmed = np.array(list(reversed(sorted([val for _, val in englishBiFreqDistStemmed.items()]))))
biThresholdCountStemmed = np.argmin(englishBiFreqListStemmed.cumsum() < englishBiFreqDistStemmed.N()*0.8)
print('Unique Count: ', len(englishBiFreqDistStemmed))
print('5 Most Frequent occurring: ', englishBiFreqDistStemmed.most_common(5))
print('Number of Bigrams required for 80% coverage after stemming: ' + str(biThresholdCountStemmed))

"""##### Bigrams required for 80% coverage - Hindi"""

hindiBiGramsStemmed = extractNGrams(hindiWordTokensStemmed, 2)
print(len(hindiBiGramsStemmed))
hindiBiFreqDistStemmed = nltk.FreqDist(hindiBiGramsStemmed)
hindiBiFreqListStemmed = np.array(list(reversed(sorted([val for _, val in hindiBiFreqDistStemmed.items()]))))
biThresholdCountStemmed = np.argmin(hindiBiFreqListStemmed.cumsum() < hindiBiFreqDistStemmed.N()*0.8)
print('Unique Count: ', len(hindiBiFreqDistStemmed))
print('5 Most Frequent occurring: ', hindiBiFreqDistStemmed.most_common(5))
print('Number of Bigrams required for 80% coverage after stemming: ' + str(biThresholdCountStemmed))

"""##### Trigrams required for 70% coverage - English"""

englishTriGramsStemmed = extractNGrams(englishWordTokensStemmed, 3)
print(len(englishTriGramsStemmed))
englishTriFreqDistStemmed = nltk.FreqDist(englishTriGramsStemmed)
englishTriFreqListStemmed = np.array(list(reversed(sorted([val for _, val in englishTriFreqDistStemmed.items()]))))
triThresholdCountStemmed = np.argmin(englishTriFreqListStemmed.cumsum() < englishTriFreqDistStemmed.N()*0.7)
print('Unique Count: ', len(englishTriFreqDistStemmed))
print('5 Most Frequent occurring: ', englishTriFreqDistStemmed.most_common(5))
print('Number of Trigrams required for 70% coverage after stemming: ' + str(triThresholdCountStemmed))

"""##### Trigrams required for 70% coverage - Hindi"""

hindiTriGramsStemmed = extractNGrams(hindiWordTokensStemmed, 3)
print(len(hindiTriGramsStemmed))
hindiTriFreqDistStemmed = nltk.FreqDist(hindiTriGramsStemmed)
hindiTriFreqListStemmed = np.array(list(reversed(sorted([val for _, val in hindiTriFreqDistStemmed.items()]))))
triThresholdCountStemmed = np.argmin(hindiTriFreqListStemmed.cumsum() < hindiTriFreqDistStemmed.N()*0.7)
print('Unique Count: ', len(hindiTriFreqDistStemmed))
print('5 Most Frequent occurring: ', hindiTriFreqDistStemmed.most_common(5))
print('Number of Trigrams required for 70% coverage after stemming: ' + str(triThresholdCountStemmed))

"""### 1.3.3 - Writing some of your basic codes and comparing with results obtained using tools

#### 1 - Analysis after implementing Heuristics

##### Utility fucntion for implementing Heuristics
"""

def getTokensAfterHeuristics(text):
    pattern = r'''(?x)          # set flag to allow verbose regexps
        (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
      | (?:\s\w\w\.)+       # Dr. Mr. Ms.
      | \w+(?:-\w+)*        # words with optional internal hyphens
      | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
      | \.\.\.              # ellipsis
      | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
    '''
    
    sentences = nltk.tokenize.RegexpTokenizer(pattern).tokenize(text)
    tokens = []
    englishTokenizer = TreebankWordTokenizer()

    for sentence in sentences:
        words = englishTokenizer.tokenize(sentence)
        tokens += words
    return tokens

def ignoreUpperCase(tokens):
    tokens = [t.lower() for t in tokens]
    return tokens

print("Tokens count Without Heuristics: ", len(englishWordTokens))
englishTokensHeur = getTokensAfterHeuristics(englishText)
englishTokensHeur = ignoreUpperCase(englishTokensHeur)
print("Tokens count With Heuristics: ", len(englishTokensHeur))

def hindiHeuristics(text):
  sentences = text.split(u"।")
  
  all_words = []
  for sentence in sentences:
    words = nltk.word_tokenize(sentence)
    all_words += words
  return all_words
# print("Tokens count Without Heuristics: ", len(hindiWordTokens))
hindiTokensHeur = hindiHeuristics(hindiText)
print("Tokens count With Heuristics: ", len(hindiTokensHeur))

"""##### DATA ANALYSIS WITHOUT STEMMING

###### Unigrams required for 90% coverage
"""

print(len(englishTokensHeur))
englishUniGramsHeur = extractNGrams(englishTokensHeur, 1)
print(len(englishUniGramsHeur))
englishUniFreqDistHeur = nltk.FreqDist(englishUniGramsHeur)
englishUniFreqListHeur = np.array(list(reversed(sorted([val for _, val in englishUniFreqDistHeur.items()]))))
uniThresholdCountHeur = np.argmin(englishUniFreqListHeur.cumsum() < englishUniFreqDistHeur.N()*0.9)
print('Number of Unigrams required for 90% coverage: ' + str(uniThresholdCountHeur))
print('Unique Count: ', len(englishUniFreqDistHeur))
print('5 Most Frequent occurring: ', englishUniFreqDistHeur.most_common(5))

"""###### Bigrams required for 80% coverage"""

print(len(englishTokensHeur))
englishBiGramsHeur = extractNGrams(englishTokensHeur, 2)
print(len(englishBiGramsHeur))
englishBiFreqDistHeur = nltk.FreqDist(englishBiGramsHeur)
englishBiFreqListHeur = np.array(list(reversed(sorted([val for _, val in englishBiFreqDistHeur.items()]))))
BiThresholdCountHeur = np.argmin(englishBiFreqListHeur.cumsum() < englishBiFreqDistHeur.N()*0.8)
print('Number of Bigrams required for 80% coverage: ' + str(BiThresholdCountHeur))
print('Unque Count: ', len(englishBiFreqDistHeur))
print('5 Most Frequent occurring: ', englishBiFreqDistHeur.most_common(5))

"""###### Trigrams required for 70% coverage"""

print(len(englishTokensHeur))
englishTriGramsHeur = extractNGrams(englishTokensHeur, 3)
print(len(englishTriGramsHeur))
englishTriFreqDistHeur = nltk.FreqDist(englishTriGramsHeur)
englishTriFreqListHeur = np.array(list(reversed(sorted([val for _, val in englishTriFreqDistHeur.items()]))))
TriThresholdCountHeur = np.argmin(englishTriFreqListHeur.cumsum() < englishTriFreqDistHeur.N()*0.7)
print('Number of Trigrams required for 70% coverage: ' + str(TriThresholdCountHeur))
print('Unique Count: ', len(englishTriFreqDistHeur))
print('5 Most Frequent occurring: ', englishTriFreqDistHeur.most_common(5))

"""##### DATA ANALYSIS WITH STEMMING"""

englishTokensHeur = [ps.stem(word) for word in englishTokensHeur]

"""###### Unigrams required for 90% coverage"""

print(len(englishTokensHeur))
englishUniGramsHeur = extractNGrams(englishTokensHeur, 1)
print(len(englishUniGramsHeur))
englishUniFreqDistHeur = nltk.FreqDist(englishUniGramsHeur)
englishUniFreqListHeur = np.array(list(reversed(sorted([val for _, val in englishUniFreqDistHeur.items()]))))
uniThresholdCountHeur = np.argmin(englishUniFreqListHeur.cumsum() < englishUniFreqDistHeur.N()*0.9)
print('Number of Unigrams required for 90% coverage: ' + str(uniThresholdCountHeur))
print('Unique Count: ', len(englishUniFreqDistHeur))
print('5 Most Frequent occurring: ', englishUniFreqDistHeur.most_common(5))

"""###### Bigrams required for 80% coverage"""

print(len(englishTokensHeur))
englishBiGramsHeur = extractNGrams(englishTokensHeur, 2)
print(len(englishBiGramsHeur))
englishBiFreqDistHeur = nltk.FreqDist(englishBiGramsHeur)
englishBiFreqListHeur = np.array(list(reversed(sorted([val for _, val in englishBiFreqDistHeur.items()]))))
BiThresholdCountHeur = np.argmin(englishBiFreqListHeur.cumsum() < englishBiFreqDistHeur.N()*0.8)
print('Number of Bigrams required for 80% coverage: ' + str(BiThresholdCountHeur))
print('Unque Count: ', len(englishBiFreqDistHeur))
print('5 Most Frequent occurring: ', englishBiFreqDistHeur.most_common(5))

"""###### Trigrams required for 70% coverage"""

print(len(englishTokensHeur))
englishTriGramsHeur = extractNGrams(englishTokensHeur, 3)
print(len(englishTriGramsHeur))
englishTriFreqDistHeur = nltk.FreqDist(englishTriGramsHeur)
englishTriFreqListHeur = np.array(list(reversed(sorted([val for _, val in englishTriFreqDistHeur.items()]))))
TriThresholdCountHeur = np.argmin(englishTriFreqListHeur.cumsum() < englishTriFreqDistHeur.N()*0.7)
print('Number of Trigrams required for 70% coverage: ' + str(TriThresholdCountHeur))
print('Unique Count: ', len(englishTriFreqDistHeur))
print('5 Most Frequent occurring: ', englishTriFreqDistHeur.most_common(5))

"""#### 2 - Likelihood Ratio Test Implementation

##### Utility function for implementing algorithm
"""

from math import log10

def getVal(k, n, x):
  temp = log10(x) * k
  temp2 = log10(1-x) * (n-k)
  temp += temp2
  return temp

def constructCollocations(bigram_dist, unigram_dist, number_tokens):
    collocation = []
    i = 0
    for bigram, freq in bigram_dist.items():
      bigramSplit = bigram.split()
      c12 = freq
      if len(bigramSplit) < 2:
        continue

      c1 = unigram_dist[bigramSplit[0]]
      c2 = unigram_dist[bigramSplit[1]]
      n = number_tokens
      if c1 == 0 or c1 == n:
        continue
      
      p = c2/n
      p1 = c12/c1
      p2 = (c2 - c12)/(n-c1)
      if(p2 <= 0 or p1 <= 0 or p <= 0):
          continue

      if(p2 >= 1 or p1 >= 1 or p >= 1):
          continue

      val = getVal(c12, c1, p) + getVal(c2 - c12, n-c1, p) - getVal(c12, c1, p1) - getVal(c2 - c12, n-c1, p2)
      val *= -2

      if(val >= 7.88):
          collocation.append(bigram)

    return collocation

"""##### Analysis for English"""

print('Type of bigrams as input:', len(englishBiFreqDist))
print('Type of unigrams as input:', len(englishUniFreqDist))
print('Number of unigrams as input:', len(englishUniGrams))
collocations = constructCollocations(englishBiFreqDist, englishUniFreqDist, len(englishUniGrams))
print('Number of collocations obtained:', len(collocations))
print('Sample Ouput:', collocations[0:20])

"""##### Analysis for Hindi"""

print('Type of bigrams as input:', len(hindiBiFreqDist))
print('Type of unigrams as input:', len(hindiUniFreqDist))
print('Number of unigrams as input:', len(hindiUnigrams))
collocations = constructCollocations(hindiBiFreqDist, hindiUniFreqDist, len(hindiUnigrams))
print('Number of collocations obtained:', len(collocations))
print('Sample Ouput:', collocations[0:20])

"""### 1.3.4 - Morphological parsing

#### 0 - PREREQUISITES - Installing required libraries
"""

import random

!sudo apt-get install python-numpy libicu-dev
!pip install PyICU polyglot pycld2 Morfessor
from polyglot.downloader import downloader
!polyglot download morph2.en
from polyglot.text import Word

# !git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
from indicnlp.morph import unsupervised_morph 
from indicnlp import common
common.INDIC_RESOURCES_PATH="/content/indic_nlp_resources"
analyzer = unsupervised_morph.UnsupervisedMorphAnalyzer('hi')

"""#### Utility function (Sampling 5 most/least frequent words)"""

def randomFreqUniGrams(freq_dist, n, m, isLeast):
  randomUniGrams = []
  if isLeast:
    freqUniGrams = freq_dist.most_common()[-n:]
    randomUniGrams = random.choices(freqUniGrams, k=m)
  else:
    freqUniGrams = freq_dist.most_common(n)
    randomUniGrams = random.choices(freqUniGrams, k=m)
  return randomUniGrams

"""#### Morphological Analysis - English

##### Most Frequent words
"""

randomUniGrams = randomFreqUniGrams(englishUniFreqDist, 100, 5, False)
randomFreqWords = []
for word, freq in randomUniGrams:
  randomFreqWords.append(word)
print(randomFreqWords)
for word in randomFreqWords:
  word = Word(word, language="en")
  print("{:<20}{}".format(word, word.morphemes))

"""##### Least Frequent words"""

randomUniGrams = randomFreqUniGrams(englishUniFreqDist, 100, 5, True)
randomWords = []
for word, freq in randomUniGrams:
  randomWords.append(word)
print(randomWords)
for word in randomWords:
  word = Word(word, language="en")
  print("{:<20}{}".format(word, word.morphemes))

"""#### Morphological Analysis - Hindi
Dependencies: hindiUniFreqDist

##### Most Frequent words
"""

randomUniGrams = randomFreqUniGrams(hindiUniFreqDist, 100, 5, False)
randomFreqWords = []
for word, freq in randomUniGrams:
  randomFreqWords.append(word)
print(randomFreqWords)
for word in randomFreqWords:
  hindiMorph = analyzer.morph_analyze(word)
  print("{:<20}{}".format(word, hindiMorph))

"""##### Least Frequent words"""

randomUniGrams = randomFreqUniGrams(hindiUniFreqDist, 100, 5, True)
randomWords = []
for word, freq in randomUniGrams:
  randomWords.append(word)
print(randomWords)
for word in randomWords:
  hindiMorph = analyzer.morph_analyze(word)
  print("{:<20}{}".format(word, hindiMorph))

"""### 1.3.5 - Sub-word Tokenization (Byte Pair Encoding)

#### Installing required libraries
"""

import re
from collections import Counter, defaultdict

"""#### BPE Algorithm Implementation for training data"""

#builds the vocab with the given freq distribution!!
def buildVocabulary(freqDist):
    vocab = nltk.FreqDist([])
    for word, freq in freqDist.items():
        w = ''
        for c in word:
            w += c + ' ' #Adding space between each character
        w += '</w>' #Adding end of word
        vocab[w] = freq
    return vocab

#returns the frequency of each pair for the vocabulary passed!
def getVocabStats(vocab):
    allPairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            allPairs[symbols[i], symbols[i+1]] += freq
    return allPairs

#Accepts the best pair and vocabulary, and this function updates the vocab and outputs the updatedVocab
def mergeVocabulary(pair, vocabIn):
    vocabOut = nltk.FreqDist([])
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for wordIn in vocabIn:
        wordOut = p.sub(''.join(pair), wordIn)
        vocabOut[wordOut] = vocabIn[wordIn]
        
    return vocabOut

#Accepts the freqDist of the corpora and returns the training encoding(list of best pairs) with default iterations as 10
def getTrainingEncodings(freqDist, numIterations=10):
    vocab = buildVocabulary(freqDist)
    trainEncodings = []

    for i in range(numIterations):
        pairs = getVocabStats(vocab)
        if not pairs:
            break
        bestPair = max(pairs, key=pairs.get)
        trainEncodings.append(bestPair)
        vocab = mergeVocabulary(bestPair, vocab)
    
    return trainEncodings, vocab

"""#### Utility functions for testing on 10 unknown words"""

#Testing functions
def mergeVocabularyTest(pair, vocabIn):
    vocabOut = []
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

    for wordIn in vocabIn:
        wordOut = p.sub(''.join(pair), wordIn)
        vocabOut.append(wordOut)
        
    return vocabOut

def tokenizeCorpus(corpus, trainEncodings):
    vocabIn = [" ".join(word) + " </w>" for word in corpus.split()]
    for pair in trainEncodings:
        vocabIn = mergeVocabularyTest(pair, vocabIn)

    return vocabIn

def helpMorpho(corpus, lang="English"):
    corpus = corpus.split()
    if lang == "English":
        for word in corpus:
            word = Word(word, language="en")
            print("{:<20}{}".format(word, word.morphemes))
    if lang == "Hindi":
        for word in corpus:
            hin_morph = analyzer.morph_analyze(word)
            print("{:<20}{}".format(word, hin_morph))

def helpBPE(corpus, pairEncoding):
    testEncodings = tokenizeCorpus(corpus, pairEncoding)
    corpus = corpus.split()
    length = len(corpus)
    for i in range(length):
        word = corpus[i]
        BPE_encoding = testEncodings[i]
        BPE_encoding = BPE_encoding.split()
        print("{:<20}{}".format(word, BPE_encoding))

def comparisons(corpus, encoding, lang="English"):
    print("Using Byte Pairing Encoding")
    helpBPE(corpus, encoding)

    print("\n\nUsing inbuilt Morphological Analysis")
    helpMorpho(corpus, lang)

"""#### COMPARISONS [BPE VS MORPHOLOGICAL] - ENGLISH"""

print("ENGLISH CORPORA TRAINING")
englishTrainEncoding, trainingVocab = getTrainingEncodings(englishUniFreqDist,500)
print("TRAINED ENCODINGS")
print(englishTrainEncoding[0:10])
print("TOP 50 MOST FREQUENT WORDS")
print(trainingVocab.most_common(50))
print("TOP 50 LEAST FREQUENT WORDS")
print(trainingVocab.most_common()[-50:])

print("ENGLISH CORPORA TESTING")
englishCorpus = "dehydrofreezing baronetised negroising nonconsequence autarkically serendipity gobbledygook scrumptious agastopia"
comparisons(englishCorpus, englishTrainEncoding, "English")

"""#### COMPARISONS [BPE VS MORPHOLOGICAL] - HINDI"""

print("tHINDI CORPORA TRAINING")
hindiTrainEncoding, hindiTrainVocab = getTrainingEncodings(hindiUniFreqDist, 500)
print("TRAINED ENCODINGS")
print(hindiTrainEncoding[0:10])
print("TOP 50 MOST FREQUENT WORDS")
print(hindiTrainVocab.most_common(50))
print("TOP 50 LEAST FREQUENT WORDS")
print(hindiTrainVocab.most_common()[-50:])


print("\n\t\t\tHINDI CORPORA TESTING")
hindiCorpus = " संपादन दस्ता असहिष्णुता प्रत्येक बंदरगाह कुलाधिपति अधिनियम आवेग अभियांत्रिकी वैतरणी"
comparisons(hindiCorpus, hindiTrainEncoding, "Hindi")