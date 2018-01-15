% Data and Systems
% Spyros Samothrakis \
  Research Fellow, IADS \
  University of Essex 
% March 14, 2017



# About

## About

* We will now turn our attention to
    * Systems
    * Big Data
    * Online visualisation
    * Web 
    * Clusters
* Until now we have done mostly algorithms (with the exception of Pandas)
* Tips and Tricks


# System tips
## PyCharm shortcuts

* Double shift - meta-shortcut!

\includegraphics[width=0.8\textwidth]{./graphics/lec9/productivity2.png}

\tiny
(From jetbrains blog)

## Jupiter/IPython notebook shortcuts


\includegraphics[trim={0 3cm 1cm 3cm},clip,width=0.8\textwidth]{./graphics/lec9/jupiter.png}

\tiny
(From stackoverflow)

## Unix

* Some basic knowledge of unix will be extremely helpful when it comes to dealing with the systems aspect 
* Windows are indeed used for data science (depening on industry)
    * But unix is almost ubiquitous in the server environment 

* `cat`
   *  ` cat A B > C`
* `head`
* `tail` / `tail -f`


## htop

CPUS, Memory, GPUs etc

\includegraphics[width=0.99\textwidth]{./graphics/lec9/htop-small.jpeg}


## Putting commands in the background

* Quite often you have long running commands that you need to run in a remote system
* `nohup <command-name> 1>out.txt 2>err.txt &`
* If command already running
    * `ctrl+z`
    * Puts command in the background
    * `disown [-h] [job-spec]`
* You can now exit the shell

## Regular expressions and crawling the web

* Collecting data online
* Parsing files
* Example: parsing IRC logs for BobBr

`find ./ -name "*" | xargs grep "BobBr" `
`cat irc.log | grep "BobBr"`


## Regular expressions

* `^` start of a line
* `$` end of a line
* `.` any character
* `*` more than zero occurrences
* `\+` more than one occurrences

\center

Let's write some grep commands

## Scrappy

\tiny

~~~python
import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"
    start_urls = [
        'http://quotes.toscrape.com/tag/humor/',
    ]

    def parse(self, response):
        for quote in response.css('div.quote'):
            yield {
                'text': quote.css('span.text::text').extract_first(),
                'author': quote.xpath('span/small/text()').extract_first(),
            }

        next_page = response.css('li.next a::attr("href")').extract_first()
        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)

~~~

# Python

## Multi-threading

* Python does not allow native multi-threading
    * Threads can improve IO performance
    * Only one CPU core is used because of GIL
* Multi-processing
    * `copy-on-write` (not on windows)
    * Harder to share state
* scikit-learn classifiers support multi-processing (`n_jobs`)
    * Not distributed
* It actually makes tensorflow/theano slower



## Joblib

\tiny

~~~python


from math import sqrt
k = [sqrt(i ** 2) for i in range(10)]
[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

from joblib import Parallel, delayed
k = Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(1000))
[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0....]


~~~


`[Parallel(n_jobs=2)]: Done    1 out of  181 | elapsed:    0.0s remaining:    4.5s
[Parallel(n_jobs=2)]: Done  198 out of 1000 | elapsed:    1.2s remaining:    4.8s
[Parallel(n_jobs=2)]: Done  399 out of 1000 | elapsed:    2.3s remaining:    3.5s
[Parallel(n_jobs=2)]: Done  600 out of 1000 | elapsed:    3.4s remaining:    2.3s
[Parallel(n_jobs=2)]: Done  801 out of 1000 | elapsed:    4.5s remaining:    1.1s
[Parallel(n_jobs=2)]: Done 1000 out of 1000 | elapsed:    5.5s finished
`


## Map

* Performs computation on each element


\tiny

~~~python
def f(x):
    return x*x

map(f, range(10))
~~~

## Multi-processing map

* Going from map to multi-processing map is trivial
* Problems with `ctrl + c`

\tiny

~~~python

from multiprocessing import Pool

def f(x):
    return x*x

pool = Pool(processes=16)    
pool.map(f, range(10))

~~~

## Filter

* Removes some elements from a list


\tiny

~~~python

number_list = range(-5, 5)
less_than_zero = list(filter(lambda x: x < 0, number_list))

~~~

## Reduce

* Performs computation on a list 
* Returns a single result
* Combines elements iteratively
* Reminds you of anything? 

\tiny

~~~python


reduce((lambda x, y: x * y), [1, 2, 3, 4])

~~~


## MapReduce

* Very commonly used paradigm for processing large datasets on multiple machines
    * Not used as much anymore
* Each machine has a piece of the data
* Map step --> Each machine applies a function to the data it has locally
* Shuffle step --> Data is redistributed to each machine according to a key
* Reduce step --> Data is reduced per key

* So basically, the same stuff you would do locally, but with a key


## Question

* Why can't you just sample? 

# Massive datasets

## Data trumps algorithms

* It is often tempting to try to find a better algorithm to solve a certain problem
* But it has been shown time and time again that one much better off by adding more data
* Problems with neat solutions are very rare, more data 
* *Physics envy* \footnote{Halevy, Alon, Peter Norvig, and Fernando Pereira. "The unreasonable effectiveness of data." IEEE Intelligent Systems 24.2 (2009): 8-12.}
    * "An informal, incomplete grammar of the English language runs over 1,700 pages"

* We are modelling human perception as much as we are modelling cars or numbers!


## Hadoop

* Hadoop Distributed File System (HDFS)
    * Splits large files and move them around different computers
    * Data lake
        * Or more like data dump? 
* Hadoop MapReduce
    * A framework for using MapReduce in hadoop
* Hadoop is java, for python you have
    * MrJob

## HDFS

* Can be used from the command line like any other programme
* `hdfs dfs <unix-like-command>`
    * `HDFS dfs -get <filename> `
    * `HDFS dfs -put <filename> `
    * `HDFS dfs -ls <filename> `
* Can accept connections remotely

## MrJob

* Hadoop is written Java
    * Has something called the streaming API to help use other languages
* MrJob was created by Yelp, to be used on Amazon clusters
    * Elastic MapReduce
    * Hadoop
* You need to have a hadoop client configures in the machine with the appropriate environment variables


`python mrjob/examples/mr_word_freq_count.py README.rst -r hadoop > counts`

## MrJob example

\tiny

~~~python

class MRWordFrequencyCount(MRJob):

    def mapper(self, _, line):
        yield "words", len(line.split())

    def reducer(self, key, values):
        yield key, sum(values)


if __name__ == '__main__':
    MRWordFrequencyCount.run()
~~~






## Spark

* MapReduce is slowing being abandoned
* HDFS still alive
* The cluster still alive
* "...using Spark on 206 EC2 machines, we sorted 100 TB of data on disk in 23 minutes"\footnote{https://databricks.com/blog/2014/11/05/spark-officially-sets-a-new-record-in-large-scale-sorting.html}
* A number of (mostly technical) speed updates over Hadoop involving memory, but most importantly
    * Does not save the results of each map operation to disk


## Spark example


\tiny

~~~python

datafile = spark.textFile("hdfs://...")
## flatmap first flattens the results of all line.split() s in the file
datafile.flatMap(lambda line: line.split())
        .map(lambda word: (word, 1))
        .reduceByKey(lambda x, y: x+y)

~~~

## Spark Dataframes

* Spark has dataframes
* Like pandas!!!
* But slightly less advanced
    * For example, they can't read command csv files
    * But of course add-ons exist
* Spark dataframes live on the cluster
    * It means that operations on them can run on multiple machines




## Spark MLib

* Spark has its own machine learning library
* That runs in a distributed fashion!
* scikit-learn is much faster
    * But your data might not fit in memory
* Avoid unless you absolutely have a really good use case
* Prefer out of core algorithms instead

## Example (from the spark tutorial)

\tiny

~~~python

#...data is somehow loaded

(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
#  Note: Use larger numTrees in practice.
#  Setting featureSubsetStrategy="auto" lets the algorithm choose.
model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=3, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
print('Test Error = ' + str(testErr))
print('Learned classification forest model:')
print(model.toDebugString())

~~~

## scikit-learn out-of-core

* Split your data into multiple files
* Read each file individually
    * Through the data into .partial_fit()
    * Not every algorithms supports this
* Use ``Dask''

\tiny

~~~python

df = dd.read_csv('my-data-*.csv')
df = dd.read_csv('hdfs:///path/to/my-data-*.csv')
df = dd.read_csv('s3://bucket-name/my-data-*.csv')

~~~

# Databases

## Databases

* Pandas can read directly from databases
* The most common pathway is to basically get the data you need from a database
    * Do the analysis locally
* Feed the data back to the database

\tiny

~~~python 

import MySQLdb
mysql_cn= MySQLdb.connect(...)
df_mysql = pd.read_sql('select USER_NAME, USER_AGE from USERS;', con=mysql_cn)    

mysql_cn.close()

~~~

## NoSQL databases

* Cassandra, MongoDB, BigTable
    * Wide column stores
* No master-slave relationship
    * Better disaster recovery
* Speed and scalability 
* If you have constant streams of data, without much structure
* E.g. chat messages
*   * and you plan on scaling to a substantial number of users

## Data pipeline

* Problem definition
* Data collection
* Data cleaning
* Data coding
* Metric selection
* Algorithm selection
* Parameter optimisation
* Post-processing 
* Deployment
* Debug



\url{https://indico.lal.in2p3.fr/event/2914/session/1/contribution/4/material/slides/0.pdf}

# Visualisation

## Bokeh

* A python package for rendering online visualisations
* Extends seaborn and renders with the style of D3.js
* Standalone capabilities as well
* Can be combined with pandas


## From seaborn to bokeh

\tiny

~~~python

import seaborn as sns

from bokeh import mpl
from bokeh.plotting import output_file, show

tips = sns.load_dataset("tips")

sns.set_style("whitegrid")

ax = sns.violinplot(x="day", y="total_bill", hue="sex",
                    data=tips, palette="Set2", split=True,
                    scale="count", inner="stick")

output_file("violin.html")

show(mpl.to_bokeh())

~~~

## IMDB movie example

* Let's move to the browser

\url{http://bokeh.pydata.org/en/latest/docs/gallery.html}

## Conclusion

* We have seen various tools that should help once you get into more "niche" scenarios
* You don't always have those massive amounts of data
* Use keyboard shortcuts
* Avoid scaling up when you don't need it
