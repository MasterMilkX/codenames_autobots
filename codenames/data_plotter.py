import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import csv


data_rows = []
columns = []
X = []

#import from the csv
with open('30exp_results.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	line = 0
	for row in csv_reader:
		if line == 0:
			columns = row
		else:
			data_rows.append(row)
		line+=1

#set plotting data
for d in data_rows:
	dlist = [int(v) for v in d[3:]]
	X.append(dlist)


fig, axs = plt.subplots(3,3)
for i in range(len(X)):
	axs[i/3,i%3].plot(x,y)




mu,std = norm.fit(X[0])
plt.hist(X[0],bins=25,density=True,alpha=0.6,color='g')
plt.show()
