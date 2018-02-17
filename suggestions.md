# Assignments

Below you will find a list of projects for CE888 - this is still a draft and I will refine the assignments as we get closer each coursework getting formally released. 




### RL and Interpretability

Modern Reinforcement Learning helps agents learn how to act using complex patterns of text, sound and video and it's slowly moving away from research and making inroads to traditional industries (e.g., creating game NPC characters). The high dimensionality of the input space makes it very hard to interpret why an agent preferred one action over another. In this project we will try to transfer some novel methods from supervised learning to Reinforcement Learning in order to interpret why agents make certain decisions. We will use already existing Atari game playing agents and try to interpret their actions profile in real time, effectively "seeing" through the agent's eyes.

**Target Journal/Conference:** IEEE Transactions on Games



**References:**

1. [Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. "" Why Should I Trust You?": Explaining the Predictions of Any Classifier." KDD (2016).](https://arxiv.org/pdf/1602.04938v3)
2. [Lipton, Zachary C., et al. "The Mythos of Model Interpretability." IEEE Spectrum (2016)](http://zacklipton.com/media/papers/mythos_model_interpretability_lipton2016.pdf)
3. [Greydanus, Sam, et al. "Visualizing and Understanding Atari Agents." arXiv preprint arXiv:1711.00138 (2017).](https://arxiv.org/pdf/1711.00138)
4. [OpenCV optical flow tutorial](https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html)
2. [LIME](https://github.com/marcotcr/lime)

**Data:** 

1. [Example Open AI gym Atari Controllers - look also in your VM](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/A3C-Gym)


**Tasks**

1. Each Atari agent perceives the world through an concatenation of 4 frames and outputs an action. Run agents in at least 6 games, in as diverse states as possible. Collect at least 3,000 observation instances per game and their associated actions and load them for further processing. You will find the code that the agent actually runs here: [common.py](https://github.com/ppwwyyxx/tensorpack/blob/master/examples/DeepQNetwork/common.py). Note that `play_one_episode(env, func, render=False)` redefines predict using a random search - disable it. You should focus on `ob` and `act` variables inside the `while True:` loop. Also note that some helpful code is: 

~~~{Python}
from PIL import Image

stacker = np.empty((84, 0, 3),dtype="uint8")

for it in range(4):
	im = Image.fromarray(s[:, :, it*3:3*(it+1)])
    q = np.asarray(im)
    stacker = np.hstack((stacker, q))

im = Image.fromarray(stacker)
im.save("game_name-" + str(t) + ".png") # you need to define (t) somewhere so that you know which part of the game you are in. 
~~~


2. Pass the observations to LIME and get an interpreted image/observation. Save the interpreted images. 
3. Calculate the optical flow between the first image in an observation and the last image.  
4. Use one of the unsupervised learning algorithms from sci-kit learn to break down your data in various segments - are there clear clusters being formed? What is in each cluster? 
5. Create a video of interpreted agent actions and upload on youtube (optional)


* * *

### Evolutionary Strategies for Domain Adaptation

It is often the case that the distribution of the source data is not the same as the target data; for example we only have labeled data examples from images of animals we took in artificial captivity conditions (*source data*), but we would like to classify animals in the wild (*target data*). We don't know the labels of the target data, so we have to learn features that fail to discriminate between source and target distributions, but are good enough to actually learn the mapping between those distributions and their labels.

**Target Journal/Conference:** IEEE Transactions on Evolutionary Computation


**References:**

1. [Ganin, Yaroslav, and Victor Lempitsky. "Unsupervised domain adaptation by backpropagation." International Conference on Machine Learning. 2015.](http://proceedings.mlr.press/v37/ganin15.html)
2. [Lipton, Zachary C., et al. "The Mythos of Model Interpretability." IEEE Spectrum (2016)](http://zacklipton.com/media/papers/mythos_model_interpretability_lipton2016.pdf)
3. [Greydanus, Sam, et al. "Visualizing and Understanding Atari Agents." arXiv preprint arXiv:1711.00138 (2017).](https://arxiv.org/pdf/1711.00138)
4. [InfoGA archive](https://github.com/ssamot/infoGA)


**Data:** 


1. [MNIST-M Dataset, Blobs and related code](https://github.com/pumpikano/tf-dann) 
2. [The office 31 Dataset](https://github.com/jindongwang/transferlearning/blob/master/doc/dataset.md#office-31)


**Tasks**

1. Download and load the above datasets in python, clearly separating the domain and source data - write down what you observe. 
2. Create a neural network that takes as input the data provided and outputs a set of features - use RELU units. 
3. Use the outputs of the random neural network to train a Random Forest. 
4. Start an evolutionary process using SNES, adapting the weights of the neural network. The score of your classifier should take into account domain adaptation; a good classifier both succeeds in achieving good performance for the source domain, while the features learned fail to discriminate between source and target domains. The score that you give back to SNES should should thus be a weighted sum between how bad a random forest fails to discriminate between source and target, while at the same time how well it does to discriminate between the different classes. 
5. Plot the loss you get at each generation and evaluate your method in both datasets.


* * *

### Genetic Programming/Auto-ML for One-Shot Learning

One of the issues of most ML algorithms is the need for copious amounts of data - neural networks are notorious for that. It might be possible to transform our data in a way that algorithms can use a very limited number of examples and still perform well. A possible method for doing this is transforming classification/regression tasks to metric learning tasks, i.e. how far away is a new data instance from ones observed already.  


**Target Journal/Conference:** IEEE Transactions on Neural Networks and Learning Systems


**References:**

1. [One Shot Learning and Siamese Networks in Keras](https://sorenbouma.github.io/blog/oneshot/)
2. [Lake, Brenden M., Ruslan Salakhutdinov, and Joshua B. Tenenbaum. "Human-level concept learning through probabilistic program induction." Science 350.6266 (2015): 1332-1338.](https://staff.fnwi.uva.nl/t.e.j.mensink/zsl2016/zslpubs/lake15science.pdf)
3. [TPOT](https://github.com/EpistasisLab/tpot)

**Data:** 


1. [Omniglot dataset](https://github.com/brendenlake/omniglot) 

**Tasks:**

1. Download and load the omniglot dataset, clearly separating the training and the test data. Generate image combinations (same/different) according the blog post above (reference 2).
2. Use TPOT to learn a classifier/regressor based on Random Forests that tries to differentiate between objects being in the same or different class. 
3. Use the classifier as a metric function for a scikit-learn `sklearn.neighbors.KNeighborsClassifier` with a custom metric function that is based on the classifier/regressor learned above. 
4. Revisit the TPOT regressor/classifier by adding `class_weight` as `balanced` or `balanced_subsample` in the parameters of the classifier/regressor.




* * *

### Continual Learning using auto-encoders

One of the most important unsolved issues in Machine Learning is learning concepts incrementally. For this project we will try a novel idea of creating auto-encoders to detect if there has been a change in the setting we find ourselves in. 

**Target Journal/Conference:** IEEE Transactions on Neural Networks and Learning Systems


**References:**

1. [Keras Autoencoder](https://blog.keras.io/building-autoencoders-in-keras.html)
1. [Kirkpatrick, James, et al. "Overcoming catastrophic forgetting in neural networks." Proceedings of the National Academy of Sciences (2017): 201611835.](http://www.pnas.org/content/114/13/3521.full)
1. [Goodfellow, Ian J., et al. "An empirical investigation of catastrophic forgetting in gradient-based neural networks." arXiv preprint arXiv:1312.6211 (2013).](https://arxiv.org/pdf/1312.6211.pdf)

**Data:** 

1. [MNIST dataset](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)
1. [CIFAR10 dataset](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py)


**Tasks**

1. Create three or more tasks of MNIST digits by permuting the pixels (i.e shuffling the images) in a fixed way for each task and save the new datasets in a file. Do the same with CIFAR10 data. 
2. Instantiate $n$ of keras/neural network autoencoders and associate an instance of a classifier with each one.
3. Send data in batches and pick the auto-encoder with the lowest error - train the autoencoder and the associated classifier with that batch.
4. Evaluate the setup in all tasks.
5. Redo the above experiment with an increased amount of autoencoders ($n+1$). 
6. Create new autoencoders on the fly if all the the already instantiated autoencoders errors are too high. 

* * * 


* * *

* * *

