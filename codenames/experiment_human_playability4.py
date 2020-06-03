import os
import subprocess
import random

#define config variables
NUM_EXP = 1

#algorithms to use (assuming in ai4games folder)
CODEMASTER = ["ai4games.codemaster_transformer_weighted_v2",
 "players.codemaster_w2vglove_05_v2 --w2v vectors/GoogleNews-vectors-negative300.bin --glove vectors/glove.6B.50d.txt", 
 "human", 
 "human"]
GUESSER = ["human", 
"human", 
"ai4games.guesser_transformer", 
"players.guesser_w2vglove --w2v vectors/GoogleNews-vectors-negative300.bin --glove vectors/glove.6B.50d.txt"]

OUTPUT_FILE = "human_exp_results.csv"		#where to print the results of each pairing

outFile = open(OUTPUT_FILE, 'w')
#os.system('rm bot_results.txt')		#remove the old bot results

#set up headers
outFile.write("codemaster,guesser,avg turns," + (",".join(str(n) for n in range(NUM_EXP))) + "," + "red,bad" + "\n")

#ask what game to play
i = int(input("Enter the game number : "))
#order = list(range(len(CODEMASTER)))
#order = random.sample(order, len(order))


#run round robin experiments x number of times
#for i in i:

turnTable = []

for t in range(NUM_EXP):
	print('Game', i)#(CODEMASTER[i] + "-" + GUESSER[i] + " " + str(t+1) + "/" + str(NUM_EXP) + "         ")
	cmd = 'python3 game_v2.py ' + CODEMASTER[i] + ' ' + GUESSER[i] #+ ' --board boards/board_' + str(t+1) + ".txt"
	
	os.system(cmd)		#runs the command with output

	#grab the counter value from the bot_results.txt file
	botFile = open("bot_results.txt", 'r')
	newRes = botFile.readlines()[-1]
	p = newRes.split(" ")
	turns = int(p[0].split(":")[1])
	print(turns)
	blue_flip = int(p[1].split(":")[1])
	civil_flip = int(p[2].split(":")[1])
	assass_flip = int(p[3].split(":")[1])
	red_flip = int(p[4].split(":")[1])
	bad_flip = blue_flip + civil_flip + assass_flip
	print (red_flip)
	print (bad_flip)
	turnTable.append(turns)

	botFile.close()


	'''
	proc = os.popen(cmd).read()
	lines = proc.split("\n")

	turns = 25
	endCt = [g for g in lines if "Game Counter:" in g]

	#print(lines[-2])

	turns = int(lines[lines.index(endCt[0])].split(": ")[1])
	print(turns)
	turnTable.append(turns)
	'''


print("")
avg = sum(turnTable) / NUM_EXP
outFile.write(CODEMASTER[i] + "," + GUESSER[i] + "," + str(avg)  + "," + ",".join(str(v) for v in turnTable) + "," + str(red_flip) + "," + str(bad_flip)+ "\n")

outFile.close()
print(" --- EXPERIMENT FINISHED --- ")
