# TERM-FREQUENCY INVERSE DOCUMENT FREQUENCY CODEMASTER
# CODE WRITTEN BY MILK


from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import gutenberg
from numpy.linalg import norm
from players.codemaster import codemaster
from operator import itemgetter
from numpy import *
import gensim.models.keyedvectors as word2vec
import gensim.downloader as api
import itertools
import numpy as np
import random
import scipy


class ai_codemaster(codemaster):

	def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
		#not necessary
		'''
		self.brown_ic = brown_ic
		self.glove_vecs = glove_vecs
		self.word_vectors = word_vectors
		self.wordnet_lemmatizer = WordNetLemmatizer()
		self.lancaster_stemmer = LancasterStemmer()
		'''
		calcTFIDF(20)

		print(self.tf_hash)
		print(self.idf_hash)

		self.cm_wordlist = []
		with open('players/cm_wordlist.txt') as infile:
			for line in infile:
				self.cm_wordlist.append(line.rstrip())

	def receive_game_state(self, words, maps):
		self.words = words
		self.maps = maps

	def give_clue(self):

		# Creates Red-Labeled Word arrays, and everything else arrays
		for i in range(25):
			if self.words[i][0] == '*':
				continue
			elif self.maps[i] == "Assassin" or self.maps[i] == "Blue" or self.maps[i] == "Civilian":
				bad_words.append(self.words[i].lower())
			else:
				red_words.append(self.words[i].lower())
		#print("RED:\t", red_words)




		return ["", 1]		#return a tuple of a string and an integer




	#TF-IDF CODE BELOW
	def calcTFIDF(self, num_books):
		#randomly select books to use
		books = gutenberg.fileids()
		sel_books = random.choices(books, k=num_books)	


		#make 2 tables of word frequencies
		self.tf_hash = {}
		self.idf_hash = {}

		#iterate through each
		for b in sel_books:
			#get the unique words from the book
			words = gutenberg.words(b)
			num_words = len(words)
			u, c = np.unique(words, return_counts=True)

			#get tf = (# times word w appears / # of words total)
			tf = {}
			for i in range(len(u)):
				tf[u[i]] = (c[i]/num_words)
			self.tf_hash[b] = tf


			#get pre-idf = (# documents with word w)
			for w in u:
				if w in self.idf:
					self.idf[w] += 1
				else
					self.idf[w] = 1

		#calculate final idf
		for w in self.idf.keys():
			self.idf[w] = np.log(num_books/self.idf[w])





