import wikipedia
import string
from PyDictionary import PyDictionary

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize


books = "categories.txt"
outputFile = "wikiDict.txt"
catSet = {}

def exportWikiDict():

	wordSet = open(books, "r").read().split(",")

	actual_dict = PyDictionary
	n = 0
	for c in wordSet:
		n+=1
		print(str(n) + "/" + str(len(wordSet)) + " categories summarized : " + c + "         ", end='\r')

		try:
			p = wikipedia.summary(c)
		except wikipedia.DisambiguationError as er:
			#if this still doesn't work the library is shit and just get the definition of the word
			try:
				#print(er.options[0:3])
				#print(e.options[0])
				p = wikipedia.summary(er.options[0], sentences=1)
			except:
				defin = actual_dict.meaning(c)
				if defin is None:
					p = c + " word"
				else:
					p = max(list(defin.values()), key=len)				#return longest definition			except wikipedia.PageError:
					if type(p) is list:
						space = " "
						p = space.join(p)
		#whatever just get the definition then
		except:
			defin = actual_dict.meaning(c)
			if defin is None:
				p = c + " word"
			else:
				p = max(list(defin.values()), key=len)			#return longest definition
				if type(p) is list:
					space = " "
					p = space.join(p)

		#print(p)

		p = p.replace('\n', " ", 1000)
		p = p.replace('\n\n', " ", 1000)
		p = p.replace('\\n', " ", 1000)
		p = p.replace('\\\\n', " ", 1000)
		p.strip()
		#print(p.encode('unicode_escape'))
		words = p.split(" ")
		words = list(map(lambda x: x.lower(), words))				#lowercase
		table = str.maketrans('', '', string.punctuation)
		words = list(map(lambda x: x.translate(table), words))		#remove punctuation
		words = list(filter(lambda x: x != "", words))				#remove empty space
		summ = " ".join(words)
		words = [word for word in word_tokenize(summ) if not word in stopwords.words() and word.isalnum()]
		catSet[c] = words


	#### EXPORT IT ####
	wr = open(outputFile, 'w')
	delim = " "
	for c in catSet.keys():
		wr.write(str(c).strip() + ":" + delim.join(catSet[c]) + "\n--\n--\n")
		
	wr.close()



exportWikiDict()