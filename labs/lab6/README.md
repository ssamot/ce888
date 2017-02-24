# Lab6

## Setting up 
* Do the following from the unix prompt of your VM
	* Go to the directory you "cloned" the module files last time
	* Do `git pull origin master' to bring the new files

* Do the following tasks using your windows share or your unix account in the VM	
	* Copy the lab files from the module directory into your own github lab directory, in "lab6" folder
	* Remove everything from the copied README.md

## ipython/jupiter

* Start ipython/jupiter by typing `ipython notebook --ip='*'`
* Start a browser and connect to `http://mlvm:8888/`
	* You will need to input the token that was provided to you when you started ipython (looks like "c3fad33a4d227d5f395f6b2ce5de34c05b2dfa0ca516b36f" (NOT THIS ONE))
* Using the web page, go to lab6

## Ipython notebooks

* Inside `lab5` you will see de.ipynb

* Create a new Ipython notebook by copying de.ipynb

## Lab Exercises 

- [ ] Use a seaborn pairplot ``sns.pairplot()'' to visualise your data
- [ ] Loop over different cluster size starting from 2 until 10, using all the features present and pick the one with the lowest silhouette score
- [ ] (Optional) Save 10 runs for each cluster size and use a seaborn pointplot [http://seaborn.pydata.org/generated/seaborn.pointplot.html](http://seaborn.pydata.org/generated/seaborn.pointplot.html) without joining the lines 
to plot the confidence intervals for the silhouette score
- [ ] Change your clusterer to ``AgglomerativeClustering'' see here for more [http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) and re-do the above experiment - what do you observe?

- [ ] Once you are done, save your changes in github
	* Go inside your lab directory and do 
      * ``git add -A -v``
      * ``git commit -m <message>``
      * ``git push origin master``
