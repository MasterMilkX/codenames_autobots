import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


cm = ["transformer", "tfidf", "naivebayes"]
g = ["transformer", "tfidf", "naivebayes"]

data = [0.9,
0.17,
0.07,
0.23,
0.27,
0,
0.17,
0.17,
0.07]

hmDat = np.zeros(shape=(len(cm),len(g)))

for v in range(len(data)):
	val = data[v]
	hmDat[int(v/3)][int(v%3)] = val

heat_map = sb.heatmap(hmDat,xticklabels=cm, yticklabels=g,annot=True)

heat_map.xaxis.set_ticks_position('top')

plt.xlabel("Codemasters")
plt.ylabel("Guessers")
plt.title('Win Rate',fontsize=20)

plt.show()