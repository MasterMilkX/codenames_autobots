import os
import subprocess

#define config variables
NUM_EXP = 30
CODEMASTER = ["players.codemaster_w2vglove_05 --w2v vectors/GoogleNews-vectors-negative300.bin --glove vectors/glove.6B.300d.txt"]
GUESSER = ["ai4games.guesser_transformer_weighted", "ai4games.guesser_tfidf", "ai4games.guesser_naivebayes"]
#ALGORITHMS = ["tfidf", "transformer"]
#ALGORITHMS = ["transformer"]
OUTPUT_FILE = "30kim_cm_weight_exp_results.csv"

outFile = open(OUTPUT_FILE, 'w')

#set up headers
outFile.write("codemaster,guesser,avg turns," + (",".join(str(n) for n in range(NUM_EXP))) + "\n")

#run round robin experiments x number of times
for i in range(len(CODEMASTER)):
	for j in range(len(GUESSER)):

		turnTable = []

		for t in range(NUM_EXP):
			print(CODEMASTER[i] + "-" + GUESSER[j] + " " + str(t+1) + "/" + str(NUM_EXP) + "         ")
			cmd = 'python3 game.py ' + CODEMASTER[i] + ' ' + GUESSER[j] + ' --board boards/board_' + str(t+1) + ".txt"
			
			#print(cmd)

			proc = os.popen(cmd).read()
			lines = proc.split("\n")

			turns = 25
			endCt = [g for g in lines if "Game Counter:" in g]

			#print(lines[-2])

			turns = int(lines[lines.index(endCt[0])].split(": ")[1])
			print(turns)
			turnTable.append(turns)


		print("")
		avg = sum(turnTable) / NUM_EXP
		outFile.write(CODEMASTER[i] + "," + GUESSER[j] + "," + str(avg)  + "," + ",".join(str(v) for v in turnTable) + "\n")

outFile.close()
print(" --- EXPERIMENT FINISHED --- ")
