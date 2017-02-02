# Lab3

## Setting up 
* Do the following from the unix prompt of your VM
	* Go to the directory you "cloned" the module files last time
	* Do `git pull origin master' to bring the new files

* Do the following tasks using your windows share or your unix account in the VM	
	* Copy the lab files from the module directory into your own github lab directory, in "lab3" folder
	* Remove everything from the copied README.md

## ipython/jupiter

* Start ipython/jupiter by typing `ipython notebook --ip='*'`
* Start a browser and connect to `http://mlvm:8888/`
	* You will need to input the token that was provided to you when you started ipython (looks like "c3fad33a4d227d5f395f6b2ce5de34c05b2dfa0ca516b36f" (NOT THIS ONE))
* Using the web page, go to lab3

## Ipython notebooks

* Inside `lab3` you will see two ipython notebooks
* Open them and see what is inside
	* facebook_regression.ipynb
	* facebook_classification.ipynb

* Creating new ipython notebook
	
* Check the dataset 
	* [https://archive.ics.uci.edu/ml/datasets/Bank+Marketing](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
	* ``bank-additional-full.csv`` in your lab directory

Attribute Information:

~~~
Input variables:
# bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone') 
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric) 
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric) 
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

~~~


## Lab Exercises
In the Ipython notebook you created

- [ ] Load the data from `bank-additional-full.csv`
- [ ] Use a classifier (anything, but `ExtraTreesClassifier` with 100 estimators is the easiest option) on the data with outcome/output variable "y"
    * Convert to dummies using `df_dummies = pd.get_dummies(df)`
    * Columns "y_no" and "duration" must be deleted - use something like `del df_copy["attribute"]` for this
    * Plot histogram of the label `y_yes`
    * Get the values and run a classifier (with outcome `y_yes`)
    * Report the results of 10-Kfold stratified cross-validation
    * Get sample importances and a confusion matrix





