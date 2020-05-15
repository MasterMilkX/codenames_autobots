# WEIGHTED TRANSFORMER CODEMASTER
# CODE BY CATALINA JARAMILLO


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import gutenberg
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

from players.codemaster import codemaster

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
		self.model = AutoModelWithLMHead.from_pretrained('gpt2')

		#get stop words and what-not
		nltk.download('popular',quiet=True)
		nltk.download('words',quiet=True)
		self.corp_words = set(nltk.corpus.words.words())

		return

	def receive_game_state(self, words, maps):
		self.words = words
		self.maps = maps

	def give_clue(self):

		# 1. GET THE RED WORDS
		count = 0
		red_words = []
		blue_words = []
		civil_words = []
		assassin_word = []        
        
# 		# Creates Red-Labeled Word arrays, and everything else arrays
# 		for i in range(25):
# 			if self.words[i][0] == '*':
# 				continue
# 			elif self.maps[i] == "Assassin" or self.maps[i] == "Blue" or self.maps[i] == "Civilian":
# 				bad_words.append(self.words[i].lower())
# 			else:
# 				red_words.append(self.words[i].lower())

#		#print("RED:\t", red_words)
###############
		# Creates Red-Labeled Word arrays, and Blue-labeled, and Civilian-labeled, and Assassin-labeled arrays
		for i in range(25):
			if self.words[i][0] == '*':
				continue
			elif self.maps[i] == "Assassin":
				assassin_word.append(self.words[i].lower())
			elif self.maps[i] == "Blue":
				blue_words.append(self.words[i].lower())
			elif self.maps[i] == "Civilian":
				civil_words.append(self.words[i].lower())
			else:
				red_words.append(self.words[i].lower())


		#print("RED:\t", red_words)


		''' WRITE NEW CODE HERE '''
		# 1. Add \u0120 in front of every word to get a better embedding
		spec_red_words = list(map(lambda w: "\u0120" + w, red_words))
###############
		spec_blue_words = list(map(lambda w: "\u0120" + w, blue_words))
		spec_civil_words = list(map(lambda w: "\u0120" + w, civil_words))
		spec_assassin_word = list(map(lambda w: "\u0120" + w, ass))


    
    
		#print(spec_red_words)
		# 2. CREATE WORD EMBEDDINGS FOR THE RED WORDS
		self.red_emb = self.word_embedding(spec_red_words)  #retrieves embedding for red_words from gpt2 layer 0 (static embedding)
		self.blue_emb = self.word_embedding(spec_blue_words)
		self.assassin_emb = self.word_embedding(spec_assassin_word)
		self.civil_emb = self.word_embedding(spec_civil_words)

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
		num_words = self.red_emb.shape[0]

		dist = np.zeros((num_words,num_words))
		for i in range(num_words):
			for j in range((self.red_emb.shape[0])):
				dist[i][j] = self.cos_sim(self.red_emb[i],self.red_emb[j])

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
		#print(subset)
		#print(red_emb[subset])
		#print(dist)
		#print(how_many)


		# 4. FIND THE CENTROID OF THE SUBSET
		center = torch.mean(self.red_emb[subset], dim=0)

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

		prepositions = open('ai4games/prepositions_etc.txt').read().splitlines() #create list with prepositions
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

		#while (tries < 5):

		# 6. WORD CLEANUP AND PARSING
		recomm = []
		#numrec = (tries-1)*1000
		for i in range((tries-1)*amt,(tries)*amt):
			recomm.append(self.tokenizer.decode((int(vecinos[1][0][i])), skip_special_tokens = True, clean_up_tokenization_spaces = True))         
		clean_words = self.cleanWords(recomm)

		#print(clean_words)
			
		return self.weightWords(clean_words,low_board)
			
			
			'''
			#7. Get the first word not in the board
			for w in clean_words:
				if w not in low_board:
					return w

			#otherwise try again
			tries+=1
			'''

		#return "??"		#i got nothing out of 5000 words


	
###### new code
	def weightWords(self, rec_words):
		# exclude words on board from the recommended list
		recomm6 = [i for i in recomm5 if i not in low_board]

		# find word embedding for recommended words
		recomm6_vec = word_embedding(list(map(lambda w: "\u0120" + w, recomm6)))

		#similarity between each recommendation and assassin
		num_recomm = recomm6_vec.shape[0]   #number of words in the embedding matrix

		sim_assassin = np.zeros((num_recomm),)
		for i in range(num_recomm):
		    sim_assassin[i] = cos_sim(recomm6_vec[i],self.assassin_emb[0])

		sim_assassin = sim_assassin.reshape(sim_assassin.shape[0],1)

		# create similarity matrix for recomm and blue words
		num_recomm = recomm6_vec.shape[0]   #number of words in the clean recommendation list
		num_blue = self.blue_emb.shape[0]   #number of words in 'subset' used for centroid


		sim_blue = np.zeros((num_recomm, num_blue))
		for i in range(num_recomm):
		    for j in range(num_blue):
			sim_blue[i][j] = cos_sim(recomm6_vec[i],self.blue_emb[j])

		# create similarity matrix for recomm and civilians words
		num_recomm = recomm6_vec.shape[0]   #number of words in the clean recommendation list
		num_civil = self.civil_emb.shape[0]   #number of words in 'subset' used for centroid


		sim_civil = np.zeros((num_recomm, num_civil))
		for i in range(num_recomm):
		    for j in range(num_civil):
			sim_civil[i][j] = cos_sim(recomm6_vec[i],self.civil_emb[j])

		#dist similarity recommendations and center
		num_recomm = recomm6_vec.shape[0]   #number of words in the embedding matrix

		sim_center = np.zeros((num_recomm))
		for i in range(num_recomm):
		    sim_center[i] = cos_sim(recomm6_vec[i],center)

		sim_center = sim_center.reshape(sim_center.shape[0],1)

		#find similarity ratio for recommended words between center and (assasin, blue, civil)
		ratio_assasin = sim_center / sim_assassin
		ratio_blue = sim_center / sim_blue
		ratio_civil = sim_center / sim_civil

		#define weights for each kind of word
		assassin_weight = 5
		blue_weight = 3
		civil_weight = 1

		#find the total ratio for each recommended word
		recomm_ratio = (ratio_assasin * assassin_weight) + \
		    (np.min(ratio_blue, axis=1).reshape(ratio_blue.shape[0],1) * blue_weight) + \
		    (np.min(ratio_civil, axis=1).reshape(ratio_civil.shape[0],1) * civil_weight)


		#print(recomm_ratio)
		
		rec_rat = {}
		for r in range(len(recomm6)):
			rec_rat[recomm6[r]] = recomm_ratio[r] 
			
		print(rec_rat)
		
		#[sorted(recomm_ratio, reverse=True).index(x) for x in recomm_ratio] #from low to high
		sorted(rec_rat.items(), key=lambda item: float(item[1]), reverse=True)
		
		print(rec_rat)
		
		return rec_rat.keys()[0]
##### not working!!!!!!!!!
