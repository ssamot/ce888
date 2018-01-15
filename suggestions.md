# Assignments

Below you will find a list of projects for CE888 - this is still a draft and I will refine the assignments as we get closer each coursework getting formally released. 




### RL and Interpretability

Modern Reinforcement Learning helps agents learn how to act using complex patterns of text, sound and video and it's slowly moving away from research and making inroads to traditional industries (e.g., creating game NPC characters). The high dimensionality of the input space makes it however very hard to interpret why an agent preferred one action over another. In this project we will try to transfer some novel methods from supervised learning to Reinforcement Learning in order to interpret why agents make certain decisions. We will use already existing Atari game playing agents and try to interpret their actions profile in real time, effectively "seeing" through the agent's eyes.

**Target Journal/Conference:** IEEE Transactions on Games



**References:**

1. [Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. "" Why Should I Trust You?": Explaining the Predictions of Any Classifier." KDD (2016).](https://arxiv.org/pdf/1602.04938v3)
2. [Lipton, Zachary C., et al. "The Mythos of Model Interpretability." IEEE Spectrum (2016)](http://zacklipton.com/media/papers/mythos_model_interpretability_lipton2016.pdf)
3. [Greydanus, Sam, et al. "Visualizing and Understanding Atari Agents." arXiv preprint arXiv:1711.00138 (2017).](https://arxiv.org/pdf/1711.00138)
4. [OpenCV optical flow tutorial](https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html)


**Data:** 

1. [Example Open AI gym Atari Controllers - look also in your VM](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/OpenAIGym)
2. [LIME](https://github.com/marcotcr/lime)

**Tasks**

1. Each Atari agent perceives the world through an concatenation of 4 frames and outputs an action. Run agents in at least 6 games, in as diverse states as possible. Collect at least 30,000 data instances per game and load them for further processing.  
2. Create a sample python programme that takes an instance, passes through the policy network and outputs an action.
3. Find a useful high level image to pass to LIME (e.g. calculate the optical flow of the sequence of four actions and output a single image with the flow in it).
4. Use one of the unsupervised learning algorithms from sci-kit learn to break down your data in various segments - are there clear clusters being formed? What is in each cluster? 
5. Create a video of agent actions. 


* * *

### Genetic Programming/Auto-ML for Domain Adaptation

It is often the case that the source data is not the same as the target data; for example we only have labeled data examples from images of animals we took in artificial captivity conditions (*source data*), but we would like to classify animals in the wild (*target data*). We don't however know the labels of the target data, so we have to learn features that fail to discriminate between source and target distributions, but are good enough to actually learn the mapping between those distributions and their labels.

**Target Journal/Conference:** IEEE Transactions on Neural Networks and Learning Systems


**References:**

1. [Ganin, Yaroslav, and Victor Lempitsky. "Unsupervised domain adaptation by backpropagation." International Conference on Machine Learning. 2015.](http://proceedings.mlr.press/v37/ganin15.html)
2. [Lipton, Zachary C., et al. "The Mythos of Model Interpretability." IEEE Spectrum (2016)](http://zacklipton.com/media/papers/mythos_model_interpretability_lipton2016.pdf)
3. [Greydanus, Sam, et al. "Visualizing and Understanding Atari Agents." arXiv preprint arXiv:1711.00138 (2017).](https://arxiv.org/pdf/1711.00138)


**Data:** 


1. [MNIST-M Dataset, Blobs and related code](https://github.com/pumpikano/tf-dann) 
2. [The office 31 Dataset](https://github.com/jindongwang/transferlearning/blob/master/doc/dataset.md#office-31)


**Tasks**

1. Download and load the above datasets in python, clearly separating the domain and source data. 
2. Use the TPOT classifier to learn good classifiers for all datasets.
3. Change the scorer of the TPOT classifier to one that takes into account domain adaptation; a good classifier both succeeds in achieving good performance for the source domain, while the features learned fail to discriminate between between source and target domains. You will need to create a scoring function with signature `scorer(estimator, X, y)`. The estimator passed is an sklearn `Pipeline` object - you need to get everything but the last part of the pipeline to transform the target data as well.
4. Evaluate your method in both datasets.


* * *

### Genetic Programming/Auto-ML for One-Shot Learning

One of the issues of most ML algorithms is the need for copious amounts of data - neural networks are notorious for that. It might be possible to transform our data in a way that algorithms that algorithms can use a very limited number of examples and still perform well. A possible method for doing this is transforming classification/regression tasks to metric learning tasks, i.e. how far away is a new data instance from ones observed already.  


**Target Journal/Conference:** IEEE Transactions on Neural Networks and Learning Systems


**References:**

1. [One Shot Learning and Siamese Networks in Keras](https://sorenbouma.github.io/blog/oneshot/)
2. [Lake, Brenden M., Ruslan Salakhutdinov, and Joshua B. Tenenbaum. "Human-level concept learning through probabilistic program induction." Science 350.6266 (2015): 1332-1338.](https://staff.fnwi.uva.nl/t.e.j.mensink/zsl2016/zslpubs/lake15science.pdf)

**Data:** 


1. [Omniglot dataset](https://github.com/brendenlake/omniglot) 

**Tasks:**

1. Download and load the above datasets in python, clearly separating the training and the test data. Generate image combinations (same /different) according the blog post above (reference 2)
2. Use the TPOT classifier to learn the metric - as a common classification with probabilities task.
3. Change the scorer of the TPOT classifier; a good scorer both succeeds in achieving good performance for the original task, while the features learned fail to discriminate between between source and target domain. You will need to create a scoring function with signature `scorer(estimator, X, y)`. The estimator passed is an sklearn Pipeline object - manipulate it to get any. Re-run TPOT with your new scorer and record the results.
4. Try to use methods for upsampling/downsampling to capture the the fact that your dataset is now imbalanced - incorporate them within TPOT's pipeline if you can.




* * *

### Continual Learning using auto-encoders

One of the most important unsolved issues in Machine Learning is learning concepts incrementally. For this project we will try 

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
6. Create new autoencoders on the fly if the error is too high. 

* * * 


* * *

* * *

