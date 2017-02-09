# Lab4

## Setting up 
* Do the following from the unix prompt of your VM
	* Go to the directory you "cloned" the module files last time
	* Do `git pull origin master' to bring the new files

* Do the following tasks using your windows share or your unix account in the VM	
	* Copy the lab files from the module directory into your own github lab directory, in "lab4" folder
	* Remove everything from the copied README.md

## ipython/jupiter

* Start ipython/jupiter by typing `ipython notebook --ip='*'`
* Start a browser and connect to `http://mlvm:8888/`
	* You will need to input the token that was provided to you when you started ipython (looks like "c3fad33a4d227d5f395f6b2ce5de34c05b2dfa0ca516b36f" (NOT THIS ONE))
* Using the web page, go to lab3

## Ipython notebooks

* Inside `lab4` you will see Bandits.ipynb

* Creating new ipython notebook

## Compare and plot Q-values



This is a rather short lab

- [ ] Keep only e-decreasing, UCB and bootstrap from your examples
- [ ] Change the reward of each function from 1,0 to 1,-1 and redo the plots
- [ ] Change the rewards of each function to 10,-10 and redo the plots
		* What do you observe? 
- [ ] Get the best bandit that you have for 10,-10 and plot the Q-values for each time step - what do you observe? 

- [ ] Once you are done, save your changes in github
	* Go inside your lab directory and do 
      * ``git add -A -v``
      * ``git commit -m <message>``
      * ``git push origin master``
