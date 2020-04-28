# NAIVE BAYES CODEMASTER
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

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import difflib

import wikipedia
import string
from PyDictionary import PyDictionary

# need to find general word classification methodology (like branching)


class ai_codemaster(codemaster):

	CATEGORIES = "ai4games/categories.txt";
	WIKI_DICT_SET = "ai4games/wikiDict.txt";

	def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
		self.num = 0

		self.getCategories()
		self.actual_dictionary = PyDictionary()
		self.wikiDict = {}
		self.readInSummaries();

		self.cm_wordlist = []
		with open('players/cm_wordlist.txt') as infile:
			for line in infile:
				self.cm_wordlist.append(line.rstrip())

		self.classifyCategories()
		self.boardSum = {}
		self.curGuesses = {}

	def get_board(self, words):
		self.words = words

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
		self.curGuesses.push(self.chooseWords(self.clue, self.num, self.words))
		resortGuesses()

		bestGuess = self.curGuesses.pop(0).split("-")[0]
		return bestGuess				#returns a string for the guess


	'''
	ALGORITHM:
	-train-
	1. get the categories
	2. get the wikipedia articles for the categories
	3. use the bag of words from the articles as the training data for the categories

	-give response-
	4. get the clue and the number
	5. calculate the probabilities of each word occuring in the category
	6. choose the highest x probability words
	'''

	#sort the guesses based on value
	def resortGuesses(self):
		#split
		sortD = {}
		for g in self.curGuesses:
			p = g.split("-")
			sortD[p[0]] = float(sortD[p[1]])

		#sort + reform
		newsort = []
		for k, v in sorted(sortD.items(), key=lambda item: item[1]):
			newsort.push(str(k) + "-" + str(v))

		self.curGuesses = newsort


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


	#tokenize the summaries for the words on the board
	def readBoard(self, boardWords):
		if(len(self.boardSum.keys()) > 0):	#already done
			return


		self.boardSum = {}
		
		actual_dict = PyDictionary
		n = 0
		for b in boardWords:
			c = b.lower()
			n+=1
			print(str(n) + "/" + str(len(boardWords)) + " board words summarized : " + c + "         ", end='\r')

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

			#clean up this hot mess
			#print(p)
			#print(p.encode('unicode_escape'))
			words = p.split(" ")
			words = list(map(lambda x: x.lower(), words))				#lowercase
			table = str.maketrans('', '', string.punctuation)
			words = list(map(lambda x: x.translate(table), words))		#remove punctuation
			words = list(filter(lambda x: x != "", words))				#remove empty space
			summ = " ".join(words)
			words = [word for word in word_tokenize(summ) if not word in stopwords.words() and word.isalnum()]
			self.boardSum[c] = words


	#P(x) - x is a word
	def wordProb(self, x):
		return self.word_cts[x] / self.trainTotal

	#P(c) - c is a category
	def categoryProb(self, c):
		return self.classifyCats[c]["TOTAL_NUM"]/self.trainTotal


	#P(x|c) - x is a word, c is a category
	def featcategoryProb(self, x, c):
		if x not in self.classifyCats[c].keys():
			return 0
		return self.classifyCats[c][x]/self.classifyCats[c]["TOTAL_NUM"]

	
	#laplace smoothing instead? alternative P(x|c)
	def laplace(self, x, c):
		m = 1									# smoothing amount (add-m)
		if x in self.classifyCats[c].keys():
			t = self.classifyCats[c][x]				# number of x's for class C
		else:
			t = 0
		s = len(self.word_cts.keys())			# possible values for x
		N = self.classifyCats[c]["TOTAL_NUM"]	# number of total Cs
		return float((t+m) / (N + (m*s)))
	

	#gets the all probabilities of a word belonging to any category c
	#P(c|x) = P(x_1|c)*P(x_2|c)*...*P(x_n|c)*P(c)
	def allCategoryProb(self, x):
		artWords = self.boardSum[x]

		#calculate the probability for x belonging to each category
		catSet = {}
		for c in self.categories:
			p = 1
			for w in artWords:
				p *= self.laplace(w,c)
				#p *= self.featcategoryProb(w,c)		#continue multiplying probabilities together
				if p == 0:		#if 0, cancel calculations
					break
			p *= self.categoryProb(c)
			catSet[c] = float(p)
		return catSet


	#get the words most related to the clue (assuming the clue is a category word)
	def chooseWords(self, clue, num, boardWords):
		catProbs = {}
		for x in boardWords:
			catProbs[x] = self.laplace(x,clue)

		outD = []
		for k, v in sorted(catProbs.items(), key=lambda item: item[1]):
			outD.push(str(k) + "-" + str(v))
			print("%s: %s" % (key, value))

		#return the top x guesses
		return outD[:num]




