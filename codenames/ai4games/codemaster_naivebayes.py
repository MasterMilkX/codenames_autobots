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

import wikipedia
import string
from PyDictionary import PyDictionary

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords',quiet=True)
from nltk.tokenize import word_tokenize

# need to find general word classification methodology (like branching)


class ai_codemaster(codemaster):

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

	def receive_game_state(self, words, maps):
		self.words = words
		self.maps = maps
		self.readBoard(words)

	def give_clue(self):
		count = 0
		red_words = []
		bad_words = []
		blue_words = []
		civ_words = []
		ass_words = []

		# Creates Red-Labeled Word arrays, and everything else arrays
		for i in range(25):
			if self.words[i][0] == '*':
				continue
			elif self.maps[i] == "Assassin":
				ass_words.append(self.words[i].lower())
			elif self.maps[i] == "Blue":
				blue_words.append(self.words[i].lower())
			elif self.maps[i] == "Civilian":
				civ_words.append(self.words[i].lower())
			else:
				red_words.append(self.words[i].lower())
		print("RED:\t", red_words)


		return self.chooseCategory(red_words, ass_words, blue_words, civ_words)
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
		self.categories = list(map(lambda x: x.strip(), self.categories))

	def readInSummaries(self):
		self.wikiDict = {}
		wd = open(self.WIKI_DICT_SET, "r").read()
		wd_lines = wd.split('\n--\n--\n')
		for l in wd_lines:
			if l.strip() == "":
				continue

			parts = l.split(":")
			'''
			if(len(parts) != 2):
				print(l)
			'''

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


	#creates the occurence matrix for probabilities
	def classifyCategories(self):
		#get bag of words
		all_words = []
		bag_of_words = {}		
		self.trainTotal = 0
		self.catSet = {}

		#import it now
		if len(list(self.wikiDict.keys())) == 0:
			n = 0
			for c in self.categories:
				n+=1
				print(str(n) + "/" + str(len(self.categories)) + " categories summarized : " + c + "         ", end='\r')

				try:
					p = wikipedia.summary(c)
				except wikipedia.DisambiguationError as er:
					#if this still doesn't work the library is shit and just get the definition of the word
					try:
						#print(er.options[0:3])
						#print(e.options[0])
						p = wikipedia.summary(er.options[0])
					except:
						defin = self.actual_dictionary.meaning(c)
						if defin is None:
							p = c + " word"
						else:
							p = max(list(defin.values()), key=len)				#return longest definition			except wikipedia.PageError:
							if type(p) is list:
								space = " "
								p = space.join(p)
				#whatever just get the definition then
				except:
					defin = self.actual_dictionary.meaning(c)
					if defin is None:
						p = c + " word"
					else:
						p = max(list(defin.values()), key=len)			#return longest definition
						if type(p) is list:
							space = " "
							p = space.join(p)

				#print(p)
				p.replace('\n', " ")
				summ = p
				summ = summ.lower()
				artWords = [word for word in word_tokenize(summ) if not word in stopwords.words() and word.isalnum()]
				self.catSet[c] = artWords
				bag_of_words[c] = artWords
				self.trainTotal += len(artWords)		#add to the whole total
				for w in artWords:
					if w not in all_words:
						all_words.append(w)

		#use the external file
		else:
			i = 0
			for c in self.wikiDict.keys():
				i+= 1
				print(str(i) + "/" + str(len(self.wikiDict.keys())) + "      ", end='\r')
				#summ = " ".join(self.wikiDict[c])
				#artWords = [word for word in word_tokenize(summ) if not word in stopwords.words() and word.isalnum()]
				artWords = self.wikiDict[c]

				self.catSet[c] = artWords
				bag_of_words[c] = artWords
				self.trainTotal += len(artWords)		#add to the whole total
				for w in artWords:
					if w not in all_words:
						all_words.append(w)

		print("IMPORTED WORD SET")

		#get the counts for the probabilities
		self.classifyCats = {}			#dict[category][word] = #; dict[category][TOTAL_NUM] = #; dict[word+"_TOTAL"] = #
		word_cts = {}
		for w in all_words:
			word_cts[w] = 0

		#get the counts for each category and word
		for c in self.categories:
			w, cts = np.unique(bag_of_words[c], return_counts=True)
			self.classifyCats[c] = {}
			for a in all_words:
				self.classifyCats[c][a] = 0		#default to 0
				if a in w:						#get the count for this word in the category article
					ind = np.where(w==a)
					self.classifyCats[c][a] = cts[ind]
					word_cts[a] += cts[ind]		#add to the word's total count

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

	#return the best category from the red words and the number
	def chooseCategory(self, red_words, a_words, b_words, c_words):
		#initalize category values
		catProbs = {}
		wordCatProbs = {}
		for c in self.categories:
			catProbs[c] = 0

		#add the red words probability
		for r in red_words:
			wordCatProbs[r] = self.allCategoryProb(r)
			for c in self.categories:
				catProbs[c] += 3.0*wordCatProbs[r][c] 

		'''
		#subtract the bad words probability * some weight
		for b in a_words:
			wordCatProbs[b] = self.allCategoryProb(b)
			for c in self.categories:
				catProbs[c] -= (3.0*wordCatProbs[b][c])

		for b in b_words:
			wordCatProbs[b] = self.allCategoryProb(b)
			for c in self.categories:
				catProbs[c] -= (2.0*wordCatProbs[b][c])

		for b in c_words:
			wordCatProbs[b] = self.allCategoryProb(b)
			for c in self.categories:
				catProbs[c] -= wordCatProbs[b][c]
		'''

		#get the best category
		bestCat = ""

		#find how many words have a probability higher than 0 for this category
		s = len(self.word_cts.keys())			# possible values for x
		N = self.classifyCats[c]["TOTAL_NUM"]	# number of total Cs
		min_lap = float(1.0/(s+N))

		#debug for contenders
		for k, v in sorted(catProbs.items(), key=lambda item: float(item[1]), reverse=True):
			if bestCat == "":
				bestCat = k
			#print("%s: %s" % (k, v))

		numWords = 0
		for r in red_words:
			if wordCatProbs[r][bestCat] > min_lap:
				numWords += 1

		#default to one at a time if no good one found
		if numWords == 0:
			allCats = self.allCategoryProb(red_words[0])
			bestCat = max(allCats, key=allCats.get)
			numWords = 1

		return [bestCat, numWords]






