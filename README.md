# Standford CS231n 2017 Summary

After watching all the videos of the famous Standford's [CS231n](http://cs231n.stanford.edu/) course that took place in 2017, i decided to take summary of the whole course to help me to remember and to anyone who would like to know about it. I've skipped some contents in some lectures as it wasn't important to me.

## Table of contents

[TOC]

## Course Info

- Website: http://cs231n.stanford.edu/

- Lectures link: https://www.youtube.com/playlist?list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk

- Number of lectures: **16**

- Course description:

  - > Computer Vision has become ubiquitous in our society, with applications in search, image understanding, apps, mapping, medicine, drones, and self-driving cars. Core to many of these applications are visual recognition tasks such as image classification, localization and detection. Recent developments in neural network (aka “deep learning”) approaches have greatly advanced the performance of these state-of-the-art visual recognition systems. This course is a deep dive into details of the deep learning architectures with a focus on learning end-to-end models for these tasks, particularly image classification. During the 10-week course, students will learn to implement, train and debug their own neural networks and gain a detailed understanding of cutting-edge research in computer vision. The final assignment will involve training a multi-million parameter convolutional neural network and applying it on the largest image classification dataset (ImageNet). We will focus on teaching how to set up the problem of image recognition, the learning algorithms (e.g. backpropagation), practical engineering tricks for training and fine-tuning the networks and guide the students through hands-on assignments and a final course project. Much of the background and materials of this course will be drawn from the [ImageNet Challenge](http://image-net.org/challenges/LSVRC/2014/index).



## Introduction to CNN for visual recognition

- A brief history of Computer vision starting from the late 1960s to 2017.
- [Imagenet](http://www.image-net.org/) is one of the biggest datasets in image classification available right now.
- Starting 2012 in the Imagenet competition, CNN (Convolutional neural networks) is always winning.
- CNN actually has been invented in 1997 by [Yann Lecun](http://ieeexplore.ieee.org/document/726791/).



## Image classification

- Image classification problem has a lot of challenges like illumination and viewpoints.
  - ![](Images/39.jpeg)
- An image classification algorithm can be solved with **K nearest neighborhood** (KNN) but it can poorly solve the problem. The properties of KNN are:
  - Hyperparameters of KNN are: k and the distance measure
  - K is the number of neighbors we are comparing to.
  - Distance measures include:
    - L2 distance (Euclidean distance)
      - Best for non coordinate points
    - L1 distance (Manhattan distance)
      - Best for coordinate points
- Hyperparameters can be optimized using Cross-validation as following (In our case we are trying tp predict K):
  1. Split your dataset into `f` folds.
  2. Given predicted hyperparameters:
     - Train your algorithm with f-1 folds and test it with the remain flood. and repeat this with every fold.
  3. Choose the hyperparameters that gives the best training values (Average over all folds)
- **Linear SVM** classifier is an option for solving the image classification problem, but the curse of dimensions makes it stop improving at some point.
- **Logistic regression** is a also a solution for image classification problem, but image classification problem is non linear!
- Linear classifiers has to run the following equation: `Y = wX + b` 
  - shape of `w` is the same as `x` and shape of `b` is 1.
- We can add 1 to X vector and remove the bias so that: `Y = wX`
  - shape of `x` is `oldX+1` and `w` is the same as `x`
- We need to know how can we get `w`'s and `b`'s that makes the classifier runs at best.



## Loss function and optimization

- In the last section we talked about linear classifier but we didn't discussed how we could **train** the parameters of that model to get best `w`'s and `b`'s.

- We need a loss function to measure how good or bad our current parameters.

  - `Loss = L[i] =(f(X[i],W),Y[i])`
  - `Loss for all = 1/N * Sum(Li(f(X[i],W),Y[i]))      # Indicates the average` 

- Then we find a way to minimize the loss function given some parameters. This is called **optimization**.

- Loss function for a linear **SVM** classifier:

  - `L[i] = Sum where all classes except the predicted class (max(0, s[j] - s[y[i]] + 1))`
  - We call this ***the hinge loss***.
  - Loss function means we are happy if the best prediction are the same as the true value other wise we give an error with 1 margin.
  - Example:
    - ![](Images/40.jpg)
    - Given this example we want to compute the loss of this image.
    - `L = max (0, 437.9 - (-96.8) + 1) + max(0, 61.95 - (-96.8) + 1) = max(0, 535.7) + max(0, 159.75) = 695.45`
    - Final loss is 695.45 which is big and reflects that the cat score needs to be the best over all classes as its the lowest value now. We need to minimize that loss.
  - Its OK for the margin to be 1. But its a hyperparameter too.

- If your loss function gives you zero, are this value is the same value for your parameter? No there are a lot of parameters that can give you best score.

- You’ll sometimes hear about people instead using the squared hinge loss SVM (or L2-SVM). that penalizes violated margins more strongly (quadratically instead of linearly). The unsquared version is more standard, but in some datasets the squared hinge loss can work better.

- We add **regularization** for the loss function so that the discovered model don't overfit the data.

  - `Loss = L = 1/N * Sum(Li(f(X[i],W),Y[i])) + lambda * R(W)`
  - Where `R` is the regularizer, and `lambda` is the regularization term.

- There are different regularizations techniques:

  - | Regularizer           | Equation                            | Comments               |
    | --------------------- | ----------------------------------- | ---------------------- |
    | L2                    | `R(W) = Sum(W^2)`                   | Sum all the W squared  |
    | L1                    | `R(W) = Sum(lWl)`                   | Sum of all Ws with abs |
    | Elastic net (L1 + L2) | `R(W) = beta * Sum(W^2) + Sum(lWl)` |                        |
    | Dropout               |                                     | No Equation            |

- Regularization prefers smaller `W`s over big `W`s.

- Regularizations is called weight decay. biases should not included in regularization.

- Softmax loss (Like linear regression but works for more than 2 classes):

  - Softmax function:

    - `A[L] = e^(score[L]) / sum(e^(score[L]), NoOfClasses)`

  - Sum of the vector should be 1.

  - Softmax loss:

    - `Loss = -logP(Y = y[i]|X = x[i])`
    - Log of the probability of the good class. We want it to be near 1 thats why we added a minus.
    - Softmax loss is called cross-entropy loss.

  - Consider this numerical problem when you are computing Softmax:

    - ```python
      f = np.array([123, 456, 789]) # example with 3 classes and each having large scores
      p = np.exp(f) / np.sum(np.exp(f)) # Bad: Numeric problem, potential blowup

      # instead: first shift the values of f so that the highest number is 0:
      f -= np.max(f) # f becomes [-666, -333, 0]
      p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer
      ```

- **Optimization**:

  - How we can optimize loss functions we discussed?
  - Strategy one:
    - Get a random parameters and try all of them on the loss and get the best loss. But its a bad idea.
  - Strategy two:
    - Follow the slope.
      - ![](Images/41.png)
      - Image [source](https://rasbt.github.io/mlxtend/user_guide/general_concepts/gradient-optimization_files/ball.png).
    - Our goal is to compute the gradient of each parameter we have.
      - **Numerical gradient**: Approximate, slow, easy to write.   (But its useful in debugging.)
      - **Analytic gradient**: Exact, Fast, Error-prone.   (Always used in practice)
    - After we compute the gradient of our parameters, we compute the gradient descent:
      - `W = W - learning_rate * W_grad`
    - learning_rate is so important hyper parameter you should get the best value of it first of all the hyperparameters.
    - stochastic gradient descent:
      - Instead of using all the date, use a mini batch of examples (32/64/128 are commonly used) for faster results.

