# Artificial-Neural-Network-I-

In our Project, named “Sentiment analysis of Movie Reviews”, our goal was to capture
people’s sentiment regarding particular topic by training Neural Networks to classify the person’s sentiment on the topic. In our case, our topic was to classify Movie Reviews, where we classify the polarity of Movie Reviews dataset at a paragraph level. We worked on whether the expressed opinion in a paragraph is positive or negative (Cui et. al, 2016). We experimented with various RNN models on the Neon deep learning framework, an open-source framework developed by Nervana Systems, in order to improve accuracy in Training data and Valid data.
Methods and Algorithms used:
Programming Language: Python
1. Backends- The backend we used for our project is NervanaCPU.
2. Datasets- For our dataset which is IMDB movie reviews dataset, Neon provides an object
classes for loading, and sometimes pre-processing the data. The online source are stored in
the “_ _init_ _” method.
3. Initializers- Initializers we used in our project includes Uniform and Glorot Uniform.
Uniform initializer is used for uniform distribution from low to high. Glorot Uniform is used
6
for uniform distribution from -k to k, where k is scaled by the input dimensions k = SQRT(6/(din + dout)), where din and dout refer to the input and output dimensions of the input tensor.
4. Optimizers- We used Adagrad (Adaptive gradient) algorithm that adapts the learning rate individually for each parameter by dividing L2 -norm of all previous gradients. Given the parameters θ, gradient ∇J, accumulating norm G, and smoothing factor ε, we use the update
Figure 3: Adagrad Equation
Where the smoothing factor epsilon prevents from dividing by zero. By adjusting the learning rate individually for each parameter, Adagrad adapts to the geometry of the error surface. Differently scaled weights have appropriately scaled update steps.
