import os
import subprocess

#define config variables
NUM_EXP = 30

#algorithms to use (assuming in ai4games folder)
CODEMASTER = ["transformer", "tfidf", "naivebayes"]
GUESSER = ["transformer", "tfidf", "naivebayes"]

OUTPUT_FILE = "30weight_exp_results.csv"		#where to print the results of each pairing

outFile = open(OUTPUT_FILE, 'w')
os.system('rm bot_results.txt')		#remove the old bot results

#set up headers
outFile.write("codemaster,guesser,avg turns," + (",".join(str(n) for n in range(NUM_EXP))) + "\n")

#run round robin experiments x number of times
for i in range(len(CODEMASTER)):
	for j in range(len(GUESSER)):

		turnTable = []

		for t in range(NUM_EXP):
			print(CODEMASTER[i] + "-" + GUESSER[j] + " " + str(t+1) + "/" + str(NUM_EXP) + "         ")
			cmd = 'python3 game.py ai4games.codemaster_' + CODEMASTER[i] + ' ai4games.guesser_' + GUESSER[j] + ' --board boards/board_' + str(t+1) + ".txt"
			
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