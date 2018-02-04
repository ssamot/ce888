# Lab4

## Setting up 
* Do the following from the unix prompt of your VM
	* Go to the directory you "cloned" the module files last time
	* Do `git pull origin master' to bring the new files

* Do the following tasks using your windows share or your unix account in the VM	
	* Copy the lab files from the module directory into your own github lab directory, in "lab4_new" folder
	* Remove everything from the copied README.md

## ipython/jupiter

* Start ipython/jupiter by typing `ipython notebook --ip='*'`
* Start a browser and connect to `http://mlvm2:8888/`
	* You will need to input the token that was provided to you when you started ipython (looks like "c3fad33a4d227d5f395f6b2ce5de34c05b2dfa0ca516b36f" (NOT THIS ONE))
* Using the web page, go to lab4_new

## Ipython notebooks

* Inside `lab4_new` you will see Rec_correct.ipynb and Rec_features.ipynb

* Create a new Ipython notebook

## Lab Exercises 

- [ ] Load the data from the file ``jester-data-1.csv''
	* The data is from [http://eigentaste.berkeley.edu/dataset/](http://eigentaste.berkeley.edu/dataset/) and it contains the ratings of 101 jokes from 24,983 users
	* You can find the jokes in the website [http://eigentaste.berkeley.edu/dataset/jester_dataset_1_joke_texts.zip](http://eigentaste.berkeley.edu/dataset/jester_dataset_1_joke_texts.zip)
- [ ] Label approx 10% of the dataset cells as 99, to denote they are part of the validation set. Keep the the actual values of the cells so you can use them later. 
- [ ] Use latent factor modeling to infer the hidden ratings of the users (they are labeled as "99" in the dataset) on the training set
- [ ] Calculate the performance of the algorithm on the validation dataset
- [ ] Change hyper-parameters (i.e. learning rates, number of iterations, number of latent factors etc) as needed so you can get good results
- [ ] Report the MSE on the test dataset

- [ ] (if you have time) Use pandas to find the best and the worst rated jokes

- [ ] Once you are done, save your changes in github
	* Go inside your lab directory and do 
      * ``git add -A -v``
      * ``git commit -m <message>``
      * ``git push origin master``
