import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

def main():
	bot_res_file = open('30exp_bot_results.txt', 'r')
	res = bot_res_file.readlines()

	data = {}
	codemasters = []
	guessers = []

	curPair = ""
	curInd = 0

	for line in res:
		parts = line.split(" ")
		#TOTAL, BLUE, CIVILLIAN, ASSASSIN, RED, CM, GUESS, TIME
		inner_data = {}
		for p in parts:
			label, val = p.split(":")
			inner_data[label] = val

		#get codemaster and guesser set and make as a pair label
		cm = inner_data["CM"].split("_")[1]
		g = inner_data["GUESSER"].split("_")[1]
		if not cm in codemasters:
			codemasters.append(cm)
		if not g in guessers:
			guessers.append(g)
		pair = cm + "-" + g

		#reset current pair
		if pair != curPair:
			curPair = pair
			curInd = 0

			#initialize arrays
			data[curPair] = {}
			data[curPair]['TOTAL'] = []
			data[curPair]['B'] = []
			data[curPair]['C'] = []
			data[curPair]['A'] = []
			data[curPair]['R'] = []
			data[curPair]['CARDS_LEFT'] = []
			data[curPair]['WINS'] = []
			data[curPair]['BAD_WORDS_GUESS'] = []

		#add the values
		data[curPair]['TOTAL'].append(int(inner_data['TOTAL']))
		data[curPair]['B'].append(int(inner_data['B']))
		data[curPair]['C'].append(int(inner_data['C']))
		data[curPair]['A'].append(int(inner_data['A']))
		data[curPair]['R'].append(int(inner_data['R']))

		#calculate the cards left on the board based on the flipped number
		flipped = int(inner_data['B'])+int(inner_data['C'])+int(inner_data['A'])+int(inner_data['R'])
		data[curPair]['CARDS_LEFT'].append((25-flipped))

		data[curPair]['WINS'].append((int(inner_data['A']) == 0))

		bad_words = int(inner_data['B'])+int(inner_data['C'])+int(inner_data['A'])
		data[curPair]['BAD_WORDS_GUESS'].append(bad_words)


	#print(list(data.keys()))
	#print(data['transformer-transformer'])
	
	#csv_out(data)

	heatmap_out(data, codemasters, guessers, 'WINS',0.0,1.0)
	heatmap_out(data, codemasters, guessers, 'CARDS_LEFT')
	heatmap_out(data, codemasters, guessers, 'R',0.0,8.0)
	heatmap_out(data, codemasters, guessers, 'BAD_WORDS_GUESS',0.0,17.0)



	
def csv_out(dat):
	print("Codemaster, Guesser, Win Rate, Average Cards Left, Average Red Words Flipped, Average Bad Words Flipped, Average Turns to Win")

	#print as csv string
	for k in dat.keys():
		#pair
		output = ""
		cg = k.split("-")
		output += (cg[0] + "," + cg[1] + ",")

		#win rate
		wr = dat[k]['WINS'].count(True)
		output += (str(round(wr/len(dat[k]['WINS']),2)) + ",")

		#avg cards left
		output += (str(round(np.mean(dat[k]['CARDS_LEFT']),2)) + ",")

		#average red words flipped
		output += (str(round(np.mean(dat[k]['R']),2)) + ",")

		#average bad words flipped
		output += (str(round(np.mean(dat[k]['BAD_WORDS_GUESS']),2)) + ",")

		#average turns to win
		winturns = []
		for t in range(len(dat[k]['TOTAL'])):
			if dat[k]['WINS'][t]:
				winturns.append(dat[k]['TOTAL'][t])
		if len(winturns) > 0:
			output += (str(round(np.mean(winturns),2)))
		else:
			output += "0"


		print(output)


def heatmap_out(dat, cm, g, col,minn=None,maxx=None):
	hmDat = np.zeros(shape=(len(cm),len(g)))

	for k in dat.keys():
		cg = k.split("-")

		val = round(np.mean(dat[k][col]),2)

		hmDat[cm.index(cg[0])][g.index(cg[1])] = val

	if minn != None or maxx != None:
		heat_map = sb.heatmap(hmDat,xticklabels=cm, yticklabels=g,annot=True,vmin=minn,vmax=maxx)
	else:
		heat_map = sb.heatmap(hmDat,xticklabels=cm, yticklabels=g,annot=True)

	heat_map.xaxis.set_ticks_position('top')

	plt.xlabel("Codemasters")
	plt.ylabel("Guessers")
	plt.title(col,fontsize=20)

	plt.show()




main()
