#generates a board
import random

f = open("game_wordpool.txt", "r")
		
if f.mode == 'r':
	temp_array = f.read().splitlines()
	words = set([])
	# if duplicates were detected and the set length is not 25 then restart
	while len(words) != 25:
		words = set([])
		for x in range(0, 25):
			random.shuffle(temp_array)
			words.add(temp_array.pop())
	words = list(sorted(words))
	random.shuffle(words)

maps = ["Red"]*8 + ["Blue"]*7 + ["Civilian"]*9 + ["Assassin"]
random.shuffle(maps)

print(words)
print(maps)