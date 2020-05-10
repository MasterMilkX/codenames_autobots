# TF-IDF GUESSER
# CODE WRITTEN BY MILK


from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import gutenberg
import nltk
from numpy.linalg import norm
from players.guesser import guesser
from operator import itemgetter
from numpy import *
import gensim.models.keyedvectors as word2vec
import gensim.downloader as api
import itertools
import numpy as np
import random
import scipy

from nltk.corpus import stopwords
nltk.download('stopwords',quiet=True)
from nltk.tokenize import word_tokenize
import difflib

import wikipedia
import string
from PyDictionary import PyDictionary

# need to find general word classification methodology (like branching)


class ai_guesser(guesser):

	CATEGORIES = "ai4games/categories.txt";
	WIKI_DICT_SET = "ai4games/wikiDict.txt";

	def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
		self.num = 0

		self.actual_dictionary = PyDictionary()

		self.cm_wordlist = []
		with open('players/cm_wordlist.txt') as infile:
			for line in infile:
				self.cm_wordlist.append(line.rstrip())

		self.boardSum = {}
		self.curGuesses = []
		self.sel_books = {}

	def get_board(self, words):
		self.words = words
		self.wikipedia_calcTFIDF(words)

	def get_clue(self, clue, num):
		self.clue = clue
		self.num = num
		print("The clue is:", clue, num, sep=" ")
		li = [clue, num]
		return li

	def keep_guessing(self, clue, board):
		return len(self.curGuesses) > 0

	def give_answer(self):
		#add the new guesses to the list of possible guesses
		self.curGuesses.extend(self.chooseWords(self.clue, self.num, self.words))
		self.reSortGuesses()

		bestGuess = self.curGuesses.pop(0).split("|")[0]
		#print(self.words)
		#self.words.remove(bestGuess.upper())
		return bestGuess				#returns a string for the guess



	#sort the guesses based on value
	def reSortGuesses(self):
		#split
		sortD = {}
		for g in self.curGuesses:
			p = g.split("|")
			sortD[str(p[0])] = float(p[1])

		#sort + reform
		newsort = []
		for k, v in sorted(sortD.items(), key=lambda item: float(item[1]), reverse=True):
			if k.upper() in self.words:
				newsort.append(str(k) + "|" + str(v))

		self.curGuesses = newsort



	#set up the tf-idf table for the words
	def wikipedia_calcTFIDF(self, completeWordSet):
		if(len(list(self.sel_books.keys())) > 0):		#already got all the data, don't need to do it again
			return;

		article_res = {}
		n = 0
		for c in completeWordSet:
			w = c.lower()
			n+=1
			print(str(n) + "/" + str(len(completeWordSet)) + " words summarized : " + w + "         ", end='\r')
			k = w

			try:
				p = wikipedia.summary(w)
			except wikipedia.DisambiguationError as er:
				#if this still doesn't work the library is shit and just get the definition of the word
				try:
					#print(er.options[0:3])
					#print(e.options[0])
					p = wikipedia.summary(er.options[0])
				except:
					defin = self.actual_dictionary.meaning(w)
					if defin is None:
						p = w + " word"
					else:
						p = max(list(defin.values()), key=len)				#return longest definition			except wikipedia.PageError:
						if type(p) is list:
							space = " "
							p = space.join(p)
			#whatever just get the definition then
			except:
				defin = self.actual_dictionary.meaning(w)
				if defin is None:
					p = w + " word"
				else:
					p = max(list(defin.values()), key=len)			#return longest definition
					if type(p) is list:
						space = " "
						p = space.join(p)

			#print(p)
			p = p.replace('\n', " ", 1000)
			p = p.replace('\\n', " ", 1000)
			p = p.replace('\n\n', " ", 1000)
			p.strip()
			words = p.split(" ")
			words = list(map(lambda x: x.lower(), words))				#lowercase
			table = str.maketrans('', '', string.punctuation)
			words = list(map(lambda x: x.translate(table), words))		#remove punctuation
			words = list(filter(lambda x: x != "", words))				#remove empty space
			summ = " ".join(words)
			words = [word for word in word_tokenize(summ) if not word in stopwords.words() and word.isalnum()]
			words.append(w)		#add the word itself in case it's not already in the set
			article_res[k] = words

		self.sel_books = article_res
		

		#print("Articles: " + str(self.sel_books))

		#make 2 tables of word frequencies
		self.tf_hash = {}						#tf -> keys = book; value = hash[word] = term freq
		self.idf_hash = {}						#idf -> keys = word; value = inverse doc freq

		#iterate through each
		for b in list(self.sel_books.keys()):
			#get the unique words from the book
			words = self.sel_books[b]
			num_words = len(words)
			u, c = np.unique(words, return_counts=True)

			#get tf = (# times word w appears / # of words total)
			tf = {}
			for i in range(len(u)):
				tf[u[i]] = (c[i]/num_words)
			self.tf_hash[b] = tf


			#get pre-idf = (# documents with word w)
			for w in u:
				if w in self.idf_hash.keys():
					self.idf_hash[w] += 1
				else:
					self.idf_hash[w] = 1

		#calculate final idf
		for w in self.idf_hash.keys():
			self.idf_hash[w] = np.log(len(self.sel_books)/self.idf_hash[w])

		#print(list(self.idf_hash.keys()))


	def chooseWords(self, clue, num, boardWords):
		bestbook = ""
		bestval = 0
		totbooks = 0


		#determine the best book to use based on the clue tf-idf value
		for b in list(self.sel_books.keys()):

			if clue not in self.tf_hash[b].keys():		#word not in the book so skip
				continue
			if clue not in self.idf_hash.keys():
				continue


			print(b + ": " + str(self.tf_hash[b][clue]) + " * " + str(self.idf_hash[clue]))
			tfidf = self.tf_hash[b][clue] * self.idf_hash[clue]
			if(tfidf > bestval):
				bestbook = b
				bestval = tfidf
				totbooks+=1

		
		#found 1 match - use the 1 book
		'''
		if totbooks == 1 and num == 1:
			outD = []
			outD.append(str(bestbook) + "|" + str(bestval))
			return outD
		'''

		#could not find a good book - bogus :/
		if bestval == 0:
			print("random selection guess")
			bestbook = random.choice(list(self.sel_books.keys()))

		print("Using book: " + bestbook)
		#print(self.sel_books[bestbook])


		#use the book to get the best board words
		wordProbs = {}
		#wordProbs[x] = 0.5		#add the word itself in case it's not in the set
		for w in boardWords:
			if "*" in w:
				continue
			x = w.lower()

			if x in self.tf_hash[bestbook].keys():
				wordProbs[x] = self.tf_hash[bestbook][x]
			else:
				wordProbs[x] = 0

		outD = []
		for k, v in sorted(wordProbs.items(), key=lambda item: float(item[1]), reverse=True):
			outD.append(str(k) + "|" + str(v))
			#print("%s: %s" % (k, v))

		#return the top x guesses
		return outD[:(num)]





