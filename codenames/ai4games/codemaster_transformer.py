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



class ai_codemaster(codemaster):


	def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
		#write any initializing code here

		# DEFINE THRESHOLD VALUE
		self.dist_threshold = 0.3



		return

	def receive_game_state(self, words, maps):
		self.words = words
		self.maps = maps

	def give_clue(self):

		# 1. GET THE RED WORDS
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


		''' WRITE NEW CODE HERE '''


		# 2. DEFINE THE THRESHOLD (DONE IN THE INITIALIZING FUNCTION)

		# 3. USE THE K NEAREST NEIGHBOR ALGORITHM 

		'''

			DISTANCE MATRIX FOR EACH VECTOR
	
				a.     b        c
			a | -   | -0.2  | 0.3  |
			b | 0.3 | -     | -0.4 |
			c | 0.1 |  0.6  |  -   |

			Choose the words that has the most neighbors within 
			the distance threshold

		'''

		# 4. FIND THE CENTROID OF THE SUBSET

		# 5. USE KNN TO FIND THE CLOSEST MATCH IN THE GPT2 MATRIX FOR THE CENTROID VECTOR

		# 6. RETURN THE WORD FROM THE GPT2 MATRIX + THE NUMBER OF THE NEIGHBORS FROM THE CLUSTER



		return ["",0]





