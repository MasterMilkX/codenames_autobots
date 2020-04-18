# TERM-FREQUENCY INVERSE DOCUMENT FREQUENCY CODEMASTER
# CODE WRITTEN BY MILK


from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import gutenberg
import nltk
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

import wikipedia
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import difflib


# need to find general word classification methodology (like branching)


class ai_codemaster(codemaster):

	CATEGORIES = "ai4games/categories.txt";

	def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
		#not necessary
		'''
		self.brown_ic = brown_ic
		self.glove_vecs = glove_vecs
		self.word_vectors = word_vectors
		self.wordnet_lemmatizer = WordNetLemmatizer()
		self.lancaster_stemmer = LancasterStemmer()
		'''

		self.getCategories()

		self.cm_wordlist = []
		with open('players/cm_wordlist.txt') as infile:
			for line in infile:
				self.cm_wordlist.append(line.rstrip())

		self.classifyCategories()

	def receive_game_state(self, words, maps):
		self.words = words
		self.maps = maps

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


		return self.chooseCategory(red_words, bad_words)
		#return ["",0]		#return a tuple of a string and an integer


	'''
	ALGORITHM:
	-train-
	1. get the categories
	2. get the wikipedia articles for the categories
	3. use the bag of words from the articles as the training data for the categories

	-give clue-
	4. get the red words
	5. get the wikipedia article for each red word
	6. for each bag of word summary get the probabilities for each category 
	7. use the category with the highest total probability
	8. use the number of words with a threshold value over the category

	'''




	def getCategories(self):
		self.categories = open(self.CATEGORIES, "r").read().split(",")


	#creates the occurence matrix for probabilities
	def classifyCategories(self):
		#get bag of words
		all_words = []
		bag_of_words = {}		
		self.trainTotal = 0
		for c in self.categories:
			try:
				summ = wikipedia.summary(c)
			except wikipedia.DisambiguationError as e:
				summ = wikipedia.summary(e.options[0])

			#lowercase and tokenize the words (bag of words)
			summ = summ.lower()
			artWords = [word for word in word_tokenize(summ) if not word in stopwords.words() and word.isalnum()]
			bag_of_words[c] = artWords
			self.trainTotal += len(artWords)		#add to the whole total
			for w in artWords:
				if w not in all_words:
					all_words.append(w)

		#get the counts for the probabilities
		self.classifyCats = {}			#dict[category][word] = #; dict[category][TOTAL_NUM] = #; dict[word+"_TOTAL"] = #
		word_cts = {}
		for w in all_words:
			word_cts[a] = 0

		#get the counts for each category and word
		for c in self.categories:
			w, cts = np.unique(bag_of_words[c], return_counts=True)
			for a in all_words:
				self.classifyCats[c][a] = 0		#default to 0
				if a in w:						#get the count for this word in the category article
					self.classifyCats[c][a] = cts[w.indexOf(a)]
					word_cts[a] += cts[w.indexOf(a)]		#add to the word's total count

			self.classifyCats[c]["TOTAL_NUM"] = len(bag_of_words[c])


		self.word_cts = word_cts


	#P(x) - x is a word
	def wordProb(self, x):
		return self.word_cts[x] / self.trainTotal

	#P(c) - c is a category
	def categoryProb(self, c):
		return self.classifyCats[c]["TOTAL_NUM"]/self.trainTotal

	#P(x|c) - x is a word, c is a category
	def featcategoryProb(self, x, c):
		return self.classifyCats[c][x]/self.classifyCats[c]["TOTAL_NUM"]

	'''
	#laplace smoothing instead? alternative P(x|c)
	def laplace(self, x, c):
		m = 1									# smoothing amount (add-m)
		t = self.classifyCats[c][x]				# number of x's for class C
		s = len(self.word_cts.keys())			# possible values for x
		N = self.classifyCats[c]["TOTAL_NUM"]	# number of total Cs
		return (t+m) / (N + (m*s))
	'''

	#gets the all probabilities of a word belonging to any category c
	#P(c|x) = P(x_1|c)*P(x_2|c)*...*P(x_n|c)*P(c)
	def allCategoryProb(self, word):
		#get the summary for the word
		try:
			summ = wikipedia.summary(c)
		except wikipedia.DisambiguationError as e:
			summ = wikipedia.summary(e.options[0])

		#lowercase and tokenize the words (bag of words) and apply them to the vector of the training data
		summ = summ.lower()
		artWords = [word for word in word_tokenize(summ) if not word in stopwords.words() and word.isalnum() and word in self.word_cts.keys()]

		#calculate the probability for x belonging to each category
		catSet = {}
		for c in self.categories:
			p = 1
			for w in artWords:
				p *= self.featcategoryProb(w,c)		#continue multiplying probabilities together
				if p == 0:		#if 0, cancel calculations
					break
			p *= categoryProb(c)
			catSet[c] = p
		return catSet

	#return the best category from the red words and the number
	def chooseCategory(self, red_words, bad_words):
		#initalize category values
		catProbs = []
		wordCatProbs = {}
		for c in self.categories:
			catProbs[c] = 0

		#add the red words probability
		for r in red_words:
			wordCatProbs[r] = self.allCategoryProb(r)
			for c in self.categories:
				catProbs[c] += wordCatProbs[r][c] 

		#subtract the bad words probability
		for b in bad_words:
			wordCatProbs[b] = self.allCategoryProb(b)
			for c in self.categories:
				catProbs -= wordCatProbs[b][c]

		#get the best category
		bestCat = max(catProbs.iteritems(), key=operator.itemgetter(1))[0]

		#find how many words have a probability higher than 0 for this category
		numWords = 0
		for r in red_words:
			if wordCatProbs[r][bestCat] > 0:
				numWords += 1

		return [bestCat, numWords]






