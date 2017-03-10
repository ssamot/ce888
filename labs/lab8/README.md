# Lab8

## Setting up 
* Do the following from the unix prompt of your VM
	* Go to the directory you "cloned" the module files last time
	* Do `git pull origin master' to bring the new files

* Do the following tasks using your windows share or your unix account in the VM	
	* Copy the lab files from the module directory into your own github lab directory, in "lab8" folder
	* Remove everything from the copied README.md



For this lab, we will train a network to do sentiment analysis on IMDB data sets - actual data from here [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/)

## Lab Exercises 

- [ ] run ``python imdb.py`` and note somewhere the test accuracy score
- [ ] Modify the code to add one more layer of 64 ``relu`` units after the embedding layer record the score (i.e. add a dense followed by an "activation" layer)
- [ ] Modify the code and add a dropout layer after the relu layer
- [ ] Remove the layers you have added previously and add a Convolution layer followed by a relu non-linearity and global max pooling (see lecture notes)
- [ ] Modify the code and add an LSTM layer in place of the convolution layer
- [ ] (Optional - and quite advanced) use both an LSTM layer and a Convolution layer and merge the results with a Merge layer
- [ ] Once you are done, save your changes in github
	* Go inside your lab directory and do 
      * ``git add -A -v``
      * ``git commit -m <message>``
      * ``git push origin master``