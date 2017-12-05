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



## 1. Introduction to CNN for visual recognition

- A brief history of Computer vision starting from the late 1960s to 2017.
- Computer vision problems includes image classification, object localization, object detection, and scene understanding.
- [Imagenet](http://www.image-net.org/) is one of the biggest datasets in image classification available right now.
- Starting 2012 in the Imagenet competition, CNN (Convolutional neural networks) is always winning.
- CNN actually has been invented in 1997 by [Yann Lecun](http://ieeexplore.ieee.org/document/726791/).



## 2. Image classification

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



## 3. Loss function and optimization

- In the last section we talked about linear classifier but we didn't discussed how we could **train** the parameters of that model to get best `w`'s and `b`'s.

- We need a loss function to measure how good or bad our current parameters.

  - ```python
    Loss = L[i] =(f(X[i],W),Y[i])
    Loss_for_all = 1/N * Sum(Li(f(X[i],W),Y[i]))      # Indicates the average
    ```

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

  - ```python
    Loss = L = 1/N * Sum(Li(f(X[i],W),Y[i])) + lambda * R(W)
    ```

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

    - ```python
      A[L] = e^(score[L]) / sum(e^(score[L]), NoOfClasses)
      ```

  - Sum of the vector should be 1.

  - Softmax loss:

    - ```python
      Loss = -logP(Y = y[i]|X = x[i])
      ```

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
      - ```python
        W = W - learning_rate * W_grad
        ```

    - learning_rate is so important hyper parameter you should get the best value of it first of all the hyperparameters.

    - stochastic gradient descent:
      - Instead of using all the date, use a mini batch of examples (32/64/128 are commonly used) for faster results.




## 4. Introduction to Neural network

- Computing the analytic gradient for arbitrary complex functions:

  - What is a Computational graphs?

    - Used to represent any function. with nodes.
    - Using Computational graphs can easy lead us to use a technique that called back-propagation. Even with complex models like CNN and RNN.

  - Back-propagation simple example:

    - Suppose we have `f(x,y,z) = (x+y)z`

    - Then graph can be represented this way:

    - ```
      X         
        \
         (+)--> q ---(*)--> f
        /           /
      Y            /
                  /
                 /
      Z---------/
      ```

    - We made an intermediate variable `q`  to hold the values of `x+y`

    - Then we have:

      - ```python
        q = (x+y)              # dq/dx = 1 , dq/dy = 1
        f = qz                 # df/dq = z , df/dz = q
        ```

    - Then:

      - ```python
        df/dq = z
        df/dz = q
        df/dx = df/dq * dq/dx = z * 1 = z       # Chain rule
        df/dy = df/dq * dq/dy = z * 1 = z       # Chain rule
        ```

  - So in the Computational graphs, we call each operation `f`. For each `f` we calculate the local gradient before we go on back propagation and then we compute the gradients in respect of the loss function using the chain rule.

  - In the Computational graphs you can split each operation to as simple as you want but the nodes will be a lot. if you want the nodes to be smaller be sure that you can compute the gradient of this node.

  - A bigger example:

    - ![](Images/01.png)
    - Hint: the back propagation of two nodes going to one node from the back is by adding the two derivatives.

  - Modularized implementation: forward/ backward API (example multiply code):

    - ```python
      class MultuplyGate(object):
        """
        x,y are scalars
        """
        def forward(x,y):
          z = x*y
          self.x = x  # Cache
          self.y = y	# Cache
          # We cache x and y because we know that the derivatives contains them.
          return z
        def backward(dz):
          dx = self.y * dz         #self.y is dx
          dy = self.x * dz
          return [dx, dy]
      ```

  - If you look at a deep learning framework you will find it follow the Modularized implementation where each class has a definition for forward and backward. For example:

    - Multiplication
    - Max
    - Plus
    - Minus
    - Sigmoid
    - Convolution

- So to define neural network as a function:

  - (Before) Linear score function: `f = Wx`
  - (Now) 2-layer neural network:    `f = W2*max(0,W1*x)` 
    - Where max is the RELU non linear function
  - (Now) 3-layer neural network:    `f = W3*max(0,W2*max(0,W1*x)`
  - And so on..

- Neural networks is a stack of some simple operation that forms complex operations.



## 5. Convolutional neural networks (CNNs)

- Neural networks history:
  - First perceptron machine was developed by Frank Rosenblatt in 1957. It was used to recognize letters of the alphabet. Back propagation wasn't developed yet.
  - Multilayer perceptron was developed in 1960 by Adaline/Madaline. Back propagation wasn't developed yet.
  - Back propagation was developed in 1986 by Rumeelhart.
  - There was a period which nothing new was happening with NN. Cause of the limited computing resources and data.
  - In [2006](www.cs.toronto.edu/~fritz/absps/netflix.pdf) Hinton released a paper that shows that we can train a deep neural network using Restricted Boltzmann machines to initialize the weights then back propagation.
  - The first strong results was in 2012 by Hinton in [speech recognition](http://ieeexplore.ieee.org/document/6296526/). And the [Alexnet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) "Convolutional neural networks" that wins the image net in 2012 also by Hinton's team.
  - After that NN is widely used in various applications.
- Convolutional neural networks history:
  - Hubel & Wisel in 1959 to 1968 experiments on cats cortex found that there are a topographical mapping in the cortex and that the neurons has hireical organization from simple to complex.
  - In 1998, Yann Lecun gives the paper [Gradient-based learning applied to document recognition](http://ieeexplore.ieee.org/document/726791/) that introduced the Convolutional neural networks. It was good for recognizing zip letters but couldn't run on a more complex examples.
  - In 2012 [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) used the same Yan Lecun architecture and won the image net challenge. The difference from 1998 that now we have a large data sets that can be used also the power of the GPUs solved a lot of performance problems.
  - Starting from 2012 there are CNN that are used for various tasks (Here are some applications):
    - Image classification.
    - Image retrieval.
      - Extracting features using a NN and then do a similarity matching.
    - Object detection.
    - Segmentation.
      - Each pixel in an image takes a label.
    - Face recognition.
    - Pose recognition.
    - Medical images.
    - Playing Atari games with reinforcement learning.
    - Galaxies classification.
    - Street signs recognition.
    - Image captioning.
    - Deep dream.
- ConvNet architectures make the explicit assumption that the inputs are images, which allows us to encode certain properties into the architecture.
- There are a few distinct types of Layers in ConvNet (e.g. CONV/FC/RELU/POOL are by far the most popular)
- Each Layer may or may not have parameters (e.g. CONV/FC do, RELU/POOL don’t)
- Each Layer may or may not have additional hyperparameters (e.g. CONV/FC/POOL do, RELU doesn’t)
- How Convolutional neural networks works?
  - A fully connected layer is a layer in which all the neurons is connected. Sometimes we call it a dense layer.
    - If input shape is `(X, M)` the weighs shape for this will be `(NoOfHiddenNeurons, X)`
  - Convolution layer is a layer in which we will keep the structure of the input by a filter that goes through all the image.
    - We do this with dot product: `W.T*X + b`. This equation uses the broadcasting technique.
    - So we need to get the values of `W` and `b`
    - We usually deal with the filter (`W`) as a vector not a matrix.
  - We call output of the convolution activation map. We need to have multiple activation map.
    - Example if we have 6 filters, here are the shapes:
      - Input image                        `(32,32,3)`
      - filter size                              `(5,5,3)`
        - We apply 6 filters. The depth must be three because the input map has depth of three.
      - Output of Conv.                 `(28,28,6)` 
        - if one filter it will be   `(28,28,1)`
      - After RELU                          `(28,28,6)` 
      - Another filter                     `(5,5,6)`
      - Output of Conv.                 `(24,24,10)`
  - It turns out that convNets learns in the first layers the low features and then the mid-level features and then the high level features.
  - After the Convnets we can have a linear classifier for a classification task.
  - In Convolutional neural networks usually we have some (Conv ==> Relu)s and then we apply a pool operation to downsample the size of the activation.
- What is stride when we are doing convolution:
  - While doing a conv layer we have many choices to make regarding the stride of which we will take. I will explain this by examples.
  - Stride is skipping while sliding. By default its 1.
  - Given a matrix with shape of `(7,7)` and a filter with shape `(3,3)`:
    - If stride is `1` then the output shape will be `(5,5)`              `# 2 are dropped`
    - If stride is `2` then the output shape will be `(3,3)`             `# 4 are dropped`
    - If stride is `3` it doesn't work.
  - A general formula would be `((N-F)/stride +1)`
    - If stride is `1` then `O = ((7-3)/1)+1 = 4 + 1 = 5`
    - If stride is `2` then `O = ((7-3)/2)+1 = 2 + 1 = 3`
    - If stride is `3` then `O = ((7-3)/3)+1 = 1.33 + 1 = 2.33`        `# doesn't work`

- In practice its common to zero pad the border.   `# Padding from both sides.`
  - Give a stride of `1` its common to pad to this equation:  `(F-1)/2` where F is the filter size
    - Example `F = 3` ==> Zero pad with `1`
    - Example `F = 5` ==> Zero pad with `2`
  - If we pad this way we call this same convolution.
  - Adding zeros gives another features to the edges thats why there are different padding techniques like padding the corners not zeros but in practice zeros works!
  - We do this to maintain our full size of the input. If we didn't do that the input will be shrinking too fast and we will lose a lot of data.
- Example:
  - If we have input of shape `(32,32,3)` and ten filters with shape is `(5,5)` with stride `1` and pad `2`
  - Output size will be `(32,32,10)`	                       `# We maintain the size.`
  - Size of parameters per filter `= 5*5*3 + 1 = 76`
  - All parameters `= 76 * 10 = 76`
- Number of filters is usually common to be to the power of 2.           `# To vectorize well.`
- So here are the parameters for the Conv layer:
  - Number of filters K.
    - Usually a power of 2.
  - Spatial content size F.
    - 3,5,7 ....
  - The stride S. 
    - Usually 1 or 2        (If the stride is big there will be a downsampling but different of pooling) 
  - Amount of Padding
    - If we want the input shape to be as the output shape, based on the F if 3 its 1, if F is 5 the 2 and so on.
- Pooling makes the representation smaller and more manageable.
- Pooling Operates over each activation map independently.
- Example of pooling is the maxpooling.
  - Parameters of max pooling is the size of the filter and the stride"
    - Example `2x2` with stride `2`                     `# Usually the two parameters are the same 2 , 2`
- Also example of pooling is average pooling.
  - In this case it might be learnable.