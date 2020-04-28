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
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelWithLMHead,  AutoTokenizer, GPT2Tokenizer, TFGPT2Model
torch.set_grad_enabled(False)
from sklearn.neighbors import NearestNeighbors




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
		# 1. GET EMBEDDING FOR RED WORDS USING GPT2

		tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
		model = AutoModelWithLMHead.from_pretrained("gpt2")
		
		def red_embedding(red_words):
			text_index = tokenizer.encode(red_words,add_prefix_space=True)
			red_emb = model.transformer.wte.weight[text_index,:]
			return red_emb

		red_emb = red_embedding(red_words)


		# 2. DEFINE THE THRESHOLD (DONE IN THE INITIALIZING FUNCTION)

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
		# cosine similarity
		def cos_sim(input1, input2):
		    cos = nn.CosineSimilarity(dim=0,eps=1e-6)
		    return cos(input1, input2)

		# create distance matrix for words
		num_words = red_emb.shape[0]

		dist = np.zeros((num_words,num_words))
		for i in range(num_words):
		    for j in range((red_emb.shape[0])):
		        dist[i][j] = cos_sim(red_emb[i],red_emb[j])

		## find the word with more neighbors within threshold

		# count number of neighbors below threshold for each word
		how_many = []
		for i in range(num_words):
		    how_many.append((dist[i] <= dist_threshold).sum())

		#max number of words
		clue_num = max(how_many)

		# find which is the word with max number of neighbors
		which = np.where(how_many == clue_num)[0][0]

		# find list of vectors in the subset
		subset = []
		subset.append(which)
		np.where(dist[which] <= dist_threshold)[0]
		for i in range(np.where(dist[which] <= dist_threshold)[0].shape[0]):
		    subset.append(np.where(dist[which] <= dist_threshold)[0][i])


		# 4. FIND THE CENTROID OF THE SUBSET
		center = torch.mean(red_emb[subset], dim=0)

		# 5. USE KNN TO FIND THE CLOSEST MATCH IN THE GPT2 MATRIX FOR THE CENTROID VECTOR
		emb_matrix = model.transformer.wte.weight
		vectors = emb_matrix.detach()
		knn = NearestNeighbors(n_neighbors=1)
		knn.fit(vectors)
		knn.kneighbors(center.reshape(1,-1))


		# 6. RETURN THE WORD FROM THE GPT2 MATRIX + THE NUMBER OF THE NEIGHBORS FROM THE CLUSTER
		clue = tokenizer.convert_ids_to_tokens(int(knn.kneighbors(center.reshape(1,-1))[1][0][0]))


		return [clue,clue_num]




