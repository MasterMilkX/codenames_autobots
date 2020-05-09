# GUESSER TRANSFORMER
# CODE BY CATALINA JARAMILLO

from players.guesser import guesser

import torch
from transformers import AutoModel, AutoModelWithLMHead,  AutoTokenizer, GPT2Tokenizer, TFGPT2Model

from sklearn.neighbors import NearestNeighbors

class guesser():
    words = 0
    clue = 0
    clues = []
    
    def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
        pass


class ai_guesser(guesser):

    def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
        torch.set_grad_enabled(False)

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = AutoModelWithLMHead.from_pretrained("gpt2")

        self.curGuesses = []

    def get_clue(self, clue, num):
        self.clue = clue
        self.num = num
        print("The clue is:", clue, num, sep=" ")
        li = [clue, num]
        return li

    def get_board(self, words):
        self.words = words

    def give_answer(self):
        clue_emb = self.word_embedding(self.spec_palabras([self.clue]))
        usable_words = [w for w in self.words if "*" not in w]
        usable_words = list(map(lambda w: w.lower(), usable_words))
        board_emb = self.word_embedding(self.spec_palabras(usable_words))


        # look (number) of nearest neighbors 
        knn = NearestNeighbors(n_neighbors = self.num)
        knn.fit(board_emb)
        vecinos = knn.kneighbors(clue_emb.reshape(1,-1))

        #add the best guesses into a pending list
        for i in range(self.num):
            guess = usable_words[vecinos[1][0][i]]
            d = vecinos[0][0][i]
            self.curGuesses.append(guess+"|"+str(d))

        #resort the guesses and choose the closest one
        self.reSortGuesses()
        while True:
            bestGuess = self.curGuesses.pop(0).split("|")[0]
            if bestGuess in usable_words:
                break
        return bestGuess                #returns a string for the guess



    def keep_guessing(self, clue, num):
        return len(self.curGuesses) > 0

    def is_valid(self, result):
        if result.upper() in self.words or result == "":
            return True
        else:
            return False




    #add "\u0120" in front of each word to improve embedding result
    def spec_palabras(self, palabras):
        spec_palabras = list(map(lambda w: "\u0120" + w, palabras))
        return spec_palabras


    #create word vectors for each word
    def word_embedding(self, palabras):
        text_index = self.tokenizer.encode(palabras,add_prefix_space=False)
        word_emb = self.model.transformer.wte.weight[text_index,:]
        return word_emb

    #sort the guesses based on value
    def reSortGuesses(self):
        #split
        sortD = {}
        for g in self.curGuesses:
            p = g.split("|")
            sortD[str(p[0])] = float(p[1])

        #sort + reform
        newsort = []
        for k, v in sorted(sortD.items(), key=lambda item: float(item[1])):
            if k.upper() in self.words:
                newsort.append(str(k) + "|" + str(v))

        self.curGuesses = newsort



