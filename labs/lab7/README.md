# Lab7

## Setting up 
* Do the following from the unix prompt of your VM
	* Go to the directory you "cloned" the module files last time
	* Do `git pull origin master' to bring the new files

* Do the following tasks using your windows share or your unix account in the VM	
	* Copy the lab files from the module directory into your own github lab directory, in "lab7" folder
	* Remove everything from the copied README.md


## Lab setup

Type the following in the command prompt (connect remotely via ``ssh mlvm@mlvm``)

* ``pip install --upgrade tensorflow``
* ``pip install --upgrade keras`` 

For this lab, we will train a neural network to learn how to perform classification of images from mnist - see here [https://www.kaggle.com/c/digit-recognizer/leaderboard](https://www.kaggle.com/c/digit-recognizer/leaderboard)

## Lab Exercises 

- [ ] run ``python mnist.py`` and note somewhere the test accuracy score
- [ ] Modify the code to add one more layer of 64 ``relu`` units and record the score
- [ ] Modify the code so that you are able to add as many layers of ``relu`` units as you want, controlled by a variable called ``n_hidden_layers``
- [ ] Add a Dropout layer with strength of 0.5
- [ ] (Optional) play around with different scores and optimise on the number of layers, trying to find the optimal hyperparameters
- [ ] Once you are done, save your changes in github
	* Go inside your lab directory and do 
      * ``git add -A -v``
      * ``git commit -m <message>``
      * ``git push origin master``