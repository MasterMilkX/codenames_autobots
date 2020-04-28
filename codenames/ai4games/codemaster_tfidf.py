# TERM-FREQUENCY INVERSE DOCUMENT FREQUENCY CODEMASTER
# CODE WRITTEN BY MILK

from players.codemaster import codemaster

from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import gutenberg
import nltk
from numpy.linalg import norm

from operator import itemgetter
from numpy import *
import gensim.models.keyedvectors as word2vec
import gensim.downloader as api
import itertools
import numpy as np
import random
import scipy

import wikipedia
from PyDictionary import PyDictionary
import string


class ai_codemaster(codemaster):

	#NUM_BOOKS = 18
	NUM_BOOKS = 18
	CATEGORIES = "ai4games/categories.txt";
	WIKI_DICT_SET = "ai4games/wikiDict.txt";

	def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
		#not necessary
		'''
		self.brown_ic = brown_ic
		self.glove_vecs = glove_vecs
		self.word_vectors = word_vectors
		self.wordnet_lemmatizer = WordNetLemmatizer()
		self.lancaster_stemmer = LancasterStemmer()
		'''

	#	self.gutenberg_calcTFIDF()

		#books = list(self.tf_hash.keys())
		#b1 = books[0]
		#print(self.tf_hash[b1].keys())
		#print(self.idf_hash)

		self.cm_wordlist = []
		self.getCategories()
		self.wikiDict = {}
		self.readInSummaries();

		with open('players/cm_wordlist.txt') as infile:
			for line in infile:
				self.cm_wordlist.append(line.rstrip())

		self.actual_dictionary = PyDictionary()
		self.sel_books = {}

	def receive_game_state(self, words, maps):
		self.words = words
		self.maps = maps
		self.wikipedia_calcTFIDF(self.categories)

	def give_clue(self):
		count = 0
		red_words = []
		bad_words = []

		# Creates Red-Labeled Word arrays, and everything else arrays
		for i in range(25):
			if self.words[i][0] == '*':
				continue
			elif self.maps[i] == "Assassin" or self.maps[i] == "Blue" or self.maps[i] == "Civilian":
				bad_words.append(self.words[i].lower())
			else:
				red_words.append(self.words[i].lower())
		print("RED:\t", red_words)


		bestbook, ct = self.getBestBook(red_words)	#get the most related book
		#print(bestbook)

		return [self.getBestWord(bestbook), ct]		#return a tuple of a string and an integer


	def getCategories(self):
		self.categories = open(self.CATEGORIES, "r").read().split(",")
		self.categories = list(map(lambda x: x.strip(), self.categories))

	def readInSummaries(self):
		self.wikiDict = {}
		wd = open(self.WIKI_DICT_SET, "r").read()
		wd_lines = wd.split('\n--\n--\n')
		for l in wd_lines:
			if l.strip() == "":
				continue

			parts = l.split(":")
			if(len(parts) != 2):
				print(l)

			c = parts[0].strip()
			s = parts[1].strip()
			self.wikiDict[c] = s.split(" ")
		

	#TF-IDF CODE BELOW
	def gutenberg_calcTFIDF(self):
		self.sel_books = gutenberg.fileids()

		print("Books: " + str(self.sel_books))

		#make 2 tables of word frequencies
		self.tf_hash = {}
		self.idf_hash = {}

		#iterate through each
		for b in self.sel_books:
			#get the unique words from the book
			words = gutenberg.words(b)
			words = list(map(lambda x: x.lower(), words))
			num_words = len(words)
			u, c = np.unique(words, return_counts=True)

			#get tf = (# times word w appears / # of words total)
			tf = {}
			for i in range(len(u)):
				tag = nltk.pos_tag([u[i]])						#use nouns only
				if('NN' in tag[0][1] or 'NP' in tag[0][1] or 'JJ' in tag[0][1] or 'VB' in tag[0][1]):
					tf[u[i]] = (c[i]/num_words)
			self.tf_hash[b] = tf


			#get pre-idf = (# documents with word w)
			for w in u:
				tag = nltk.pos_tag([w])
				if('NN' in tag[0][1] or 'NP' in tag[0][1] or 'JJ' in tag[0][1] or 'VB' in tag[0][1]):		#use nouns only
					if w in self.idf_hash:
						self.idf_hash[w] += 1
					else:
						self.idf_hash[w] = 1

		#calculate final idf
		for w in self.idf_hash.keys():
			self.idf_hash[w] = np.log(len(self.NUM_BOOKS)/self.idf_hash[w])

	def wikipedia_calcTFIDF(self, completeWordSet):
		if(len(list(self.sel_books.keys())) > 0):		#already got all the data, don't need to do it again
			return;

		#if didn't import the set beforehand, do it now
		if len(list(self.wikiDict.keys())) == 0:
			article_res = {}
			n = 0
			for w in completeWordSet:
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
				p.replace('\n', " ")
				words = p.split(" ")
				words = list(map(lambda x: x.lower(), words))				#lowercase
				table = str.maketrans('', '', string.punctuation)
				words = list(map(lambda x: x.translate(table), words))		#remove punctuation
				words = list(filter(lambda x: x != "", words))				#remove empty space
				article_res[k] = words

			self.sel_books = article_res
		
		#otherwise use the external set
		else:
			self.sel_books = self.wikiDict

		#print("Articles: " + str(self.sel_books))

		#make 2 tables of word frequencies
		self.tf_hash = {}
		self.idf_hash = {}

		#iterate through each
		for b in list(self.sel_books.keys()):
			#get the unique words from the book
			words = self.sel_books[b]
			'''
			words = list(map(lambda x: x.lower(), words))				#lowercase
			table = str.maketrans('', '', string.punctuation)
			words = list(map(lambda x: x.translate(table), words))		#remove punctuation
			words = list(filter(lambda x: x != "", words))				#remove empty space
			'''
			num_words = len(words)
			u, c = np.unique(words, return_counts=True)

			#get tf = (# times word w appears / # of words total)
			tf = {}
			for i in range(len(u)):
				tag = nltk.pos_tag([u[i]])						#use nouns only
				if('NN' in tag[0][1] or 'NP' in tag[0][1] or 'JJ' in tag[0][1] or 'VB' in tag[0][1]):
					tf[u[i]] = (c[i]/num_words)
			self.tf_hash[b] = tf


			#get pre-idf = (# documents with word w)
			for w in u:
				tag = nltk.pos_tag([w])
				if('NN' in tag[0][1] or 'NP' in tag[0][1] or 'JJ' in tag[0][1] or 'VB' in tag[0][1]):		#use nouns only
					if w in self.idf_hash:
						self.idf_hash[w] += 1
					else:
						self.idf_hash[w] = 1

		#calculate final idf
		for w in self.idf_hash.keys():
			self.idf_hash[w] = np.log(len(self.sel_books)/self.idf_hash[w])

		#print(list(self.idf_hash.keys()))


	def getBestBook(self, words):
		#get idfs
		idfs = []
		for w in words:
			if w in self.idf_hash:
				idfs.append(self.idf_hash[w])
			else:
				idfs.append(0)

		#calc tf-idfs for red words
		tfidfs = []
		books = list(self.sel_books.keys())
		for b in books:
			#get tfs
			bt = []
			bookset = self.tf_hash[b]
			word_keys = bookset.keys()
			for w in words:
				if w in word_keys:
					bt.append(bookset[w])
				else:
					bt.append(0)
			
			#calculate tf-idf
			hehe = []
			for i in range(len(words)):
				hehe.append(bt[i]*idfs[i])
			tfidfs.append(hehe)


		#debug
		#for t in tfidfs:
		#	print(t)

		#get the largest sum
		sums = []
		for b in range(len(books)):
			sums.append(sum(tfidfs[b]))

		m = sums.index(max(sums))
		return books[m], np.count_nonzero(tfidfs[m])

	#get the word with the best tf-idf score for a book
	def getBestWord(self, book):
		words = list(self.tf_hash[book].keys())

		tfidf = []
		for w in words:
			tfidf.append(self.tf_hash[book][w])

		
		b = tfidf.index(max(tfidf))
		return words[b]

