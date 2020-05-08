# TRANSFORMER CODEMASTER
# CODE BY CATALINA JARAMILLO


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import gutenberg
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

from numpy.linalg import norm
from players.codemaster import codemaster
from operator import itemgetter
from numpy import *
import gensim.models.keyedvectors as word2vec
import gensim.downloader as api
import itertools

import random
import scipy
import re

import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors


from transformers import AutoModel, AutoModelWithLMHead,  AutoTokenizer, GPT2Tokenizer, TFGPT2Model



class ai_codemaster(codemaster):


	def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
		#write any initializing code here

		# DEFINE THRESHOLD VALUE
		self.dist_threshold = 0.3



		# 1. GET EMBEDDING FOR RED WORDS USING GPT2
		torch.set_grad_enabled(False)
		self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
		self.model = AutoModelWithLMHead.from_pretrained("gpt2")

		#get stop words and what-not
		nltk.download('popular')
		nltk.download('words')
		self.corp_words = set(nltk.corpus.words.words())

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
		# 1. Add \u0120 in front of every word to get a better embedding
		spec_red_words = list(map(lambda w: "\u0120" + w, red_words))


		print(spec_red_words)
		# 2. CREATE WORD EMBEDDINGS FOR THE RED WORDS
		red_emb = self.word_embedding(spec_red_words)  #retrieves embedding for red_words from gpt2 layer 0 (static embedding)


		# 3. USE THE K NEAREST NEIGHBOR -LIKE ALGORITHM (FIND K NEIGHBORS BASED ON THRESHOLD)

		'''

			DISTANCE MATRIX FOR EACH VECTOR
	
				a.     b        c
			a | -   | -0.2  | 0.3  |
			b | 0.3 | -     | -0.4 |
			c | 0.1 |  0.6  |  -   |

			Choose the words that has the most neighbors within 
			the distance threshold

		'''
		

		# create distance matrix for words
		num_words = red_emb.shape[0]

		dist = np.zeros((num_words,num_words))
		for i in range(num_words):
			for j in range((red_emb.shape[0])):
				dist[i][j] = self.cos_sim(red_emb[i],red_emb[j])

		## find the word with more neighbors within threshold

		# count number of neighbors below threshold for each word
		how_many = []
		for i in range(num_words):
		    how_many.append((dist[i] >= self.dist_threshold).sum())

		#max number of words
		clue_num = max(how_many)

		# find which is the word with max number of neighbors
		donde = np.where(how_many == clue_num)[0][0]

		# find list of vectors in the subset
		subset = []
		np.where(dist[donde] >= self.dist_threshold)[0]
		for i in range(np.where(dist[donde] >= self.dist_threshold)[0].shape[0]):
		    subset.append(np.where(dist[donde] >= self.dist_threshold)[0][i])


		#DEBUG
		print(subset)
		#print(red_emb[subset])
		print(dist)
		print(how_many)


		# 4. FIND THE CENTROID OF THE SUBSET
		center = torch.mean(red_emb[subset], dim=0)

		# 5. USE KNN TO FIND THE CLOSEST MATCH IN THE GPT2 MATRIX FOR THE CENTROID VECTOR
		emb_matrix = self.model.transformer.wte.weight
		self.vectors = emb_matrix.detach()
		

		# 6. RETURN THE WORD FROM THE GPT2 MATRIX + THE NUMBER OF THE NEIGHBORS FROM THE CLUSTER
		clue = self.getBestCleanWord(center, self.words)
		

		# 6. RETURN THE WORD FROM THE GPT2 MATRIX + THE NUMBER OF THE NEIGHBORS FROM THE CLUSTER
		#clue = tokenizer.convert_ids_to_tokens(int(knn.kneighbors(center.reshape(1,-1))[1][0][0]))


		return [clue,clue_num]


	#create word vectors for each word
	def word_embedding(self, red_words):
		text_index = self.tokenizer.encode(red_words,add_prefix_space=False)
		word_emb = self.model.transformer.wte.weight[text_index,:]
		return word_emb

	# cosine similarity
	def cos_sim(self, input1, input2):
		cos = nn.CosineSimilarity(dim=0,eps=1e-6)
		return cos(input1, input2)

	#clean up the set of words
	def cleanWords(self, embed):
		recomm = [i.lower() for i in embed]
		recomm2 = ' '.join(recomm)

		recomm3 = [w for w in nltk.wordpunct_tokenize(recomm2) \
		if w.lower() in self.corp_words or not w.isalpha()]

		prepositions = open('prepositions_etc.txt').read().splitlines() #create list with prepositions
		stop_words = nltk.corpus.stopwords.words('english')		#change set format
		stop_words.extend(prepositions)					#add prepositions and similar to stopwords
		word_tokens = word_tokenize(' '.join(recomm3)) 
		recomm4 = [w for w in word_tokens if not w in stop_words]

		excl_ascii = lambda s: re.match('^[\x00-\x7F]+$', s) != None		#checks for ascii only
		is_uni_char = lambda s: (len(s) == 1) == True						#check if a univode character
		recomm5 = [w for w in recomm4 if excl_ascii(w) and not is_uni_char(w) and not w.isdigit()]

		return recomm5

	def getBestCleanWord(self, center, board):
		tries = 1
		amt = 100
		maxTry = 5

		knn = NearestNeighbors(n_neighbors=(maxTry*amt))
		knn.fit(self.vectors)
		vecinos = knn.kneighbors(center.reshape(1,-1))

		low_board = list(map(lambda w: w.lower(), board))

		while (tries < 5):

			# 6. WORD CLEANUP AND PARSING
			recomm = []
			#numrec = (tries-1)*1000
			for i in range((tries-1)*amt,(tries)*amt):
				recomm.append(self.tokenizer.decode((int(vecinos[1][0][i])), skip_special_tokens = True, clean_up_tokenization_spaces = True))         
			clean_words = self.cleanWords(recomm)

			print(clean_words)

			#7. Get the first word not in the board
			for w in clean_words:
				if w not in low_board:
					return w

			#otherwise try again
			tries+=1

		return "??"		#i got nothing out of 5000 words



