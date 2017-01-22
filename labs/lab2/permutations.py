import matplotlib
matplotlib.use('Agg')

import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np 
import numpy.random as npr


def permutation_resampling(num_samples, case, control):
    """Returns p-value that statistic for case is different
    from statistc for control."""

    observed_diff = abs(np.mean(case) - np.mean(control))
    num_case = len(case)

    combined = np.concatenate([case, control])
    diffs = []
    for i in range(num_samples):
        xs = npr.permutation(combined)
        diff = np.mean(xs[:num_case]) - np.mean(xs[num_case:])
        diffs.append(diff)

    pval = (np.sum(diffs > observed_diff))/float(num_samples)
    return pval


# def permutation_resampling(iterations, new, old):
# 	#data =  pd.concat([old,new], ignore_index = True).values
# 	data = np.concatenate([new,old])
# 	print data.shape
# 	#samples  = np.random.choice(data,replace = True, size = [iterations, len(data)])

# 	samples = []

# 	for i in range(iterations):
# 		s = np.random.permutation(data)
# 		samples.append(s)


# 	data_mean = (new.mean() - old.mean())
	
# 	vals = []
# 	for sample in samples:
# 		sample_new, sample_old = sample[:len(new)], sample[len(new):] 
	
# 		sta = (sample_new.mean() - sample_old.mean() )

# 		if (sta > data_mean):
# 			higher = 1.0
# 		else:
# 			higher = 0.0
# 		vals.append(higher)
# 	b = np.array(vals)
# 	#print b.mean()
# 	return b.mean()
	




if __name__ == "__main__":
	df = pd.read_csv('./vehicles.csv')
	#print df
	new =  df[df.columns[0]].dropna().values
	old =  df[df.columns[1]].dropna().values
	
	
	#new = np.array([100.0,102]*5)
	#old = np.array([100.0,99]*20)
	# old = np.array([0,0,0,0,0,0,1,0,0,1,0])
	# new = np.array([1,0,0,1,1,1,0,0,0,1,0])

	old = np.array([0,1,1,1,0,1,1,0,0,1,0])
	new = np.array([0,1,1,0,1,1,0,1,1,1,0,0,1,1,1,1,1,1,1])


	print old.mean(), new.mean(), len(old), len(new),  new.mean() - old.mean()
	#exit()
	boots = []
	for i in range(500,20000,100):
		boot = permutation_resampling(i, new, old)
		print i,boot
		boots.append([i,boot])


	df_boot = pd.DataFrame(boots,  columns=['Boostrap iterations','p-value'])
	sns_plot = sns.lmplot(df_boot.columns[0],df_boot.columns[1], data=df_boot, fit_reg=False)


	sns_plot.axes[0,0].set_xlim(0,)
	sns_plot.savefig("permutations.png",bbox_inches='tight')
	sns_plot.savefig("permutations.pdf",bbox_inches='tight')

	
	
	
	#print ("Mean: %f")%(np.mean(data))
	#print ("Var: %f")%(np.var(data))
	


	