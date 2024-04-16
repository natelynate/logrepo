---
layout: post
title:  "Understanding the Internal Structure of Transformers"
date:   2024-04-14 19:15:16 +0900
categories: study
tags: theory machinelearning
---

<h2> Transformer Structure</h2>

keywords: Embeedding, Attention, MLPs, Unembedding

![two_models]({{"/assets/images/2024-04-14-Transformers_0.PNG/" |  relative_url}})

<h4>Overview</h4>
The aim of transformer, in terms of its inputs and outputs, is when given a sequence of tokens as an input to derive a vector containing probable likelihoods of all tokens of being the of being the next token. 

All tokens are represented as a vector in a N-dimensional space. 

The core capability of the Attention module in Transformers is to allow the "relationship" between tokens to be encoded and reflected in deciding the next token. 
During the process. the meaning of each token (the geometrical properties of the representative vector of a token) will be able to be altered, to encode more fine-grained and nuanced semantics of natural languages.

Transformers have a structure characterized by a set of alternating Attention blocks and standard MLPs. In the end the aim is to derive a single vector, 
that is expected to encode all semantic subtleties of the input sequence.  

If the next token is derived, then the appended sequence can again be fed to the transformers and the same process could be repeated as long as the set context window to continuously derive the follow-up sequences.

Since the last vector is a probability vector, the sum of all elements will have to add up to 1.0. For this purpose, softmax function is used. 

Typical Softmax is $\frac{e^x_i}{\Sigma{e^z_j}}$, where is e is a vector of an arbitrary size. 

In GPT, there's <i>additional Temperature constant</i>, $T$ - embedded at both the denominator and the numerator. This determines the pattern of dominance in the resulting vector. $T$ defaults to 1.0 and this would result in no modifications. 

All logits in the vector are divided by T, and when T > 1.0, it effectively reduces the variance among the logits. 

Below is a code snippet to demonstrate the effect of temperature:

```python
import numpy as np
from math import exp

def softmax(v):
    v = np.array([exp(z) for z in v])
    v /= sum(v)
    return v

def variance(v):
    return ((v - v.mean()) ** 2).mean()
```

```python
# Case when T = 1.0
V = np.array([1.0, 10.0, 5.0, -2.0, 6.0, 7.0, 12.0, 3.0])
T = 1.0
V /= T
print("V when T=1.0:", V)
d
soft_V = softmax(V)
print("Softmax when T=1.0:", soft_V)
print("Variance when T=1.0:", variance(soft_V))
```
![outcome1]({{"/assets/images/2024-04-14-Transformers_code0.PNG/" |  relative_url}})

```python
# Case when T = 2.0, variance is reduced
V = np.array([1.0, 10.0, 5.0, -2.0, 6.0, 7.0, 12.0, 3.0])
T = 2.0
V /= T # Apply temperature

print()
print()
print("V when T=2.0:", V)

soft_V = softmax(V)
print("Softmax when T=2.0:", soft_V)
print("Variance when T=2.0:", variance(soft_V))
```

![outcome1]({{"/assets/images/2024-04-14-Transformers_code1.PNG/" |  relative_url}})

```python
# Case when T = 0.5, variance is increased
V = np.array([1.0, 10.0, 5.0, -2.0, 6.0, 7.0, 12.0, 3.0])
T = 0.5
V /= T # Apply temperature

print()
print()
print("V when T=0.5:", V)

soft_V = softmax(V)
print("Softmax when T=0.5:", soft_V)
print("Variance when T=0.5:", variance(soft_V))
```

![outcome1]({{"/assets/images/2024-04-14-Transformers_code2.PNG/" |  relative_url}})
<br>
<br>
<br>

More uniform the resulting softmax is, it widens the likelihood of different tokens to be selected. This creates cognitive impression to the human readers that LLM outcomes are more "creative" with its selection of words. 

<br>
<br>

