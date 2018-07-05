# cond-shift-neurons

Implementation of Conditionally Shifted Neurons (CSN) from:

[Munkhdalai, T., et al. **"Rapid adaptation with conditionally shifted neurons."** *Proceedings of the 35th International Conference on Machine Learning*. 2018.](https://arxiv.org/pdf/1712.09926.pdf)

*The author's implementation of CSN seems to be unreleased as of yet and in any case, the author mentioned the code is in Chainer. So here's an implementation in Tensorflow!*

## Summary

Here, Munkhdalai et al. built on their previous work ([Munkhdalai et al., 2017](https://arxiv.org/abs/1703.00837)) and introduced a adaptive-at-test-time network architecture for metalearning.

In regular neural network training, we calculate the gradients of the loss function with respect to the network parameters. Then the gradients are used to update the network parameters, which hopefully reduces test loss.

By aggregating parameter updates across many training iterations and samples, we ideally end up with a network that works well on any sample from the same distribution.

What if we only have an extremely small dataset? For instance, in a 5-way 1-shot task, we only have 5 training samples, 1 per class. We could train the network by simply aggregating network updates across the 5 samples. **An interesting alternative explored here is to decide which training samples are the most relevant during test time and only update the network using the most relevant training samples.**

Or to cite the paper: 
> Additionally, [conditionally shifted neurons] have the capacity to shift their
activation values on the fly based on auxiliary conditioning information. These conditional shifts adapt model behavior to the task at hand.

Briefly, here's what happens during metatest:

*Assume a 3-way 1-shot classification task with a main classifier network. The initialization of the main classifier network has been metatrained but has not seen the new metatest task.*

1. Each training sample is used to generate a Key-Value pair, where the Key is an embedding of the sample and the Value is the set of corresponding network updates ie. using Value of sample 1 to update the classifier is akin to training the classifier on sample 1
2. Each test sample is used to generate a Key, which is then compared against the Training Keys, with an attention/alignment mechanism
3. Calculate a set of network updates using the alignment of the Test Key with the Training Keys ie. if the alignment is 80%-10%-10% then take 80% of Value 1 + 10% of Value 2 + 10% of Value 3
4. *Shift* the main classifier network using the calculated set of network updates (termed as Conditionally Shifted Neurons)
5. Classify the test sample with the shifted network

In other words, if the test sample is most aligned to training sample 1, we use gradient information derived from training sample 1 to update the network and classify the test sample.

**If the test sample is most aligned to training sample 1, why don't we just classify it to be the same class as training sample 1?**

Well, we can do that, but consider the case where the test sample is equally aligned to all three samples. Then the algorithm automatically incorporates gradient information from all three samples to make the classification. If we just use the alignment to make the decision, we would be stuck.

## Interesting Notes and Relations

*These are my observations and not mentioned by the authors ie. any mistakes are entirely on my part.*

**1-step Gradient Descent**

In the case that gradient information from 1 training sample is used, the shifted network is akin to the initial network plus one step of gradient descent on the single training sample.

Likewise, if gradient information from all training samples are used, the shifted network is akin to the initial network plus one step of gradient descent on the entire training set as a single minibatch.

CSN can hence be seen as a general form of these two cases.

**Relation to Learned Initializations**

During metatraining, the main network is also trained such that the initial network plus a one-step *"shift"* during test time should be sufficient to classify the test sample correctly.

Hence, the metatraining of CSN also optimizes the parameters/*"initialization"* of the initial network, similar to MAML by Finn et al. ([2017](https://arxiv.org/abs/1703.03400)) and Reptile by Nichol et al. ([2018](https://arxiv.org/abs/1803.02999)). 

Considered as a more general form of MAML, CSN parameterizes the network update function, whereas MAML/Reptile uses a regular gradient descent algorithm. (But using a parameterized update for MAML/Reptile is definitely possible.)

**Relation to Learned Optimizers**

Andrychowicz et al. ([2016](https://arxiv.org/abs/1606.04474)) and Ravi & Larochelle ([2017](https://openreview.net/pdf?id=rJY0-Kcll)) introduced parameterized optimizers, which allowed us to optimized optimizers (hohoho!). 

Similarly, the Value Function here is a sort of parameterized optimizer that takes in gradient information and outputs network updates ie. the *"shifts"*. The algorithm also incorporates training and test information to output the final set of CSNs for classification. This is akin to a dynamic optimizer that uses gradient information and sample context to output an optimized update.

**Relation to Metric-based Metalearners**

As mentioned earlier, it could be possible to use the calculated alignment between test and training samples to directly output the classification.

This would then be similar to Matching Networks by Vinyals et al. ([2016](https://arxiv.org/abs/1606.04080)) and Prototypical Networks by Snell et al. ([2017](https://arxiv.org/abs/1703.05175)), as well as the use of Siamese networks in Koch's thesis ([2015](http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf)).

However, instead of classifying based on a similarity metric, the similarity/alignment here is used to provide information for updating a classification network. This might be arguably more robust than metric/similarity-based metalearners, although less efficient.














