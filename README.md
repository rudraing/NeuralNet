# NeuralNet
implementing a simple neural network for classification using numpy and python
The provided code appears to be a partial implementation of a neural network with two hidden layers, involving the following key components and functions:

Initialization of Parameters: The init_params function initializes the weights (w1 and w2) and biases (b1 and b2) for two hidden layers of the neural network. These parameters are randomly initialized with small values.

Activation Functions: The code defines two activation functions:

ReLU(z): This is the Rectified Linear Unit (ReLU) activation function, which applies an element-wise operation to introduce non-linearity.
softmax(z): This function calculates the softmax of the input z, typically used for multiclass classification problems to obtain class probabilities.
Forward Propagation: The forward_prop function calculates the forward pass of the neural network. It takes input features X and computes the following steps:

Compute the weighted sum and apply the ReLU activation for the first hidden layer.
Compute the weighted sum and apply the softmax activation for the second hidden layer.
Return the weighted sums (z1 and z2) and the activations (a1 and a2) for both hidden layers.
One-Hot Encoding: The one_hot function converts class labels Y into a one-hot encoded format. This is commonly used in multiclass classification tasks to represent class labels as binary vectors.

Derivative of ReLU: The deri_ReLU function calculates the derivative of the ReLU activation function, which is used during backpropagation.

Backpropagation: The back_prop function computes the gradients of the weights and biases during backpropagation. It uses the chain rule to compute gradients for the two hidden layers.

Update Parameters: The update_params function updates the weights and biases using the computed gradients and a learning rate (alpha).

Training (Gradient Descent): The grad_des function is intended for training the neural network. It performs gradient descent by iteratively updating the parameters. It also prints the accuracy of the model every 50 iterations.

Main Training Code: The main part of the code appears to be training the neural network using the grad_des function with some input data (X_train and Y_train) for a specified number of iterations and a learning rate.

The code aims to implement a simple neural network for classification, but it may require further refinement, error handling, and additional components to be a complete and functional neural network. Additionally, there are some issues that need to be resolved, such as the shape mismatch error mentioned earlier, which should be addressed for the code to run correctly.
