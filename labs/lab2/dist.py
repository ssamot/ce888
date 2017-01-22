import matplotlib
matplotlib.use('Agg')

import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import uniform, pareto, norm

#mean, var, skew, kurt = 
b = 1.0
dists = []
dists+=[[("pareto"),pareto.stats(2, moments='mvsk'), pareto]]
dists+=[[("uniform"),uniform.stats(moments='mvsk'), uniform]]
dists+=[[("normal"),norm.stats(moments='mvsk'),norm]]
dists+=[[("normal_sc"),norm.stats(moments='mvsk'),norm]]

print (dists)

size = 20000

for dist in dists:

	print dist[0]
	if(dist[0] == "pareto"):
		sample = dist[2].rvs(b, size = size)
		sample = sample[(sample < 8)]
	if(dist[0] == "normal"):
		sample = dist[2].rvs(size = size)
	if(dist[0] == "uniform"):
		sample = dist[2].rvs(size = size)
	if(dist[0] == "normal_sc"):
		print "normal high variance"
		#sample = dist[2].rvs(size = size, scale = 3)
		sample = np.random.normal(0, 3, size)
		sample = sample[((sample < 5) & (sample > -5))]


	sns_plot = sns.distplot(sample, bins = 20, kde=True, rug=True).get_figure()


	if (dist[0] == "pareto"):
		axes = plt.gca()
		axes.set_xlim([1,7])

	if (dist[0] == "uniform"):
		axes = plt.gca()
		axes.set_xlim([0,1])

	if (dist[0] == "normal_sc"):
		axes = plt.gca()
		axes.set_xlim([-5,5])

	axes = plt.gca()
	axes.set_xlabel('X') 
	axes.set_ylabel('P(X)')
	

	plt.savefig("./dists/example_" + str(dist[0]) + ".png",bbox_inches='tight')
	plt.savefig("./dists/example_" + str(dist[0]) + ".pdf",bbox_inches='tight')
	plt.clf()
