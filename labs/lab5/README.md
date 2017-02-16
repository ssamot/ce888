# Lab5

## Setting up 
* Do the following from the unix prompt of your VM
	* Go to the directory you "cloned" the module files last time
	* Do `git pull origin master' to bring the new files

* Do the following tasks using your windows share or your unix account in the VM	
	* Copy the lab files from the module directory into your own github lab directory, in "lab5" folder
	* Remove everything from the copied README.md

## ipython/jupiter

* Start ipython/jupiter by typing `ipython notebook --ip='*'`
* Start a browser and connect to `http://mlvm:8888/`
	* You will need to input the token that was provided to you when you started ipython (looks like "c3fad33a4d227d5f395f6b2ce5de34c05b2dfa0ca516b36f" (NOT THIS ONE))
* Using the web page, go to lab5

## Ipython notebooks

* Inside `lab5` you will see Rec_correct.ipynb and Rec_features.ipynb

* Create a new Ipython notebook

## Lab Exercises 

- [ ] Load the data from the file ``jester-data-1.csv''
	* The data is from [http://eigentaste.berkeley.edu/dataset/](http://eigentaste.berkeley.edu/dataset/) and it contains the ratings of 101 jokes from 24,983 users
	* You can find the jokes in the website [http://eigentaste.berkeley.edu/dataset/jester_dataset_1_joke_texts.zip](http://eigentaste.berkeley.edu/dataset/jester_dataset_1_joke_texts.zip)

- [ ] Split the data into validation, test and training set with 80:10:10 proportions
- [ ] Use latent factor modelling to infer the hidden ratings of the users (they are labelled as "99" in the dataset) on the training set
- [ ] Calculate the performance of the algorithm in the validation dataset by looping through the dataset without training
- [ ] Change hyper-parameters (i.e. learning rates, number of iterations etc) as needed so you can get good results
- [ ] Report the MSE on the test dataset

- [ ] (if you have time) Use pandas to find the best and the worst rated jokes

- [ ] Once you are done, save your changes in github
	* Go inside your lab directory and do 
      * ``git add -A -v``
      * ``git commit -m <message>``
      * ``git push origin master``
