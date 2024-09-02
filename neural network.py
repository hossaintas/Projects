import numpy as np
import nnfs
import matplotlib.pyplot as plt
import timeit

# Initialise a random seed and ensure that the dot product takes the same data type
nnfs.init()


class Layer:
    def __init__(self, n_inputs, n_neurons, weight_lambda_l2=0, bias_lambda_l2=0):
        # initialise random weights between -1 and 1
        self.bias_lambda_l2 = bias_lambda_l2
        self.weight_lambda_l2 = weight_lambda_l2
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # The forward function calculates the output
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    # uses the derivative
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dweights += 2 * self.weights * self.weight_lambda_l2
        self.dbiases += 2 * self.biases * self.dbiases
        self.dinputs = np.dot(dvalues, self.weights.T)


class ReLu:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)
        self.inputs = inputs

    def backward(self, dvalues):
        # As we need to tweak the original variable, let's make a copy of the values first
        self.dinputs = dvalues.copy()
        # If the input values are negative, make the gradient 0
        self.dinputs[self.inputs <= 0] = 0


# Activation function applied to output layer to return probabilities of each output
class SoftMax:
    def forward(self, inputs):
        bounded_outputs = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_values = np.exp(bounded_outputs)
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    # dvalues is the CE loss gradient w.r.t the inputs
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate((zip(self.output, dvalues))):
            single_output = single_output.reshape(-1, 1)
            jacobean_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # sample-wise gradient dotted with loss function
            self.dinputs[inputs] = np.dot(jacobean_matrix, single_dvalues)


# Generic loss function
class Loss:
    # y is target values, output is predicted values from the network
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        batch_loss = np.mean(sample_losses)
        return batch_loss

    def lambda_loss(self, layer):
        lambda_loss = 0
        lambda_loss += layer.weight_lambda_l2 * np.sum(np.square(layer.weights))
        lambda_loss += layer.bias_lambda_l2 * np.sum(np.square(layer.biases))
        return lambda_loss


# Ths type of Loss function takes the negative log of the predicted probability
class LossCategoricalCrossEntropy(Loss):
    def forward(self, yPred, yTrue):
        samples = len(yPred)
        # clip to avoid taking the log of 0
        yPredClipped = np.clip(yPred, 1e-7, 1 - 1e-7)
        # Takes the probability predicted for the correct class for each batch item
        # accounts for whether yTrue is given in scalar or one-hot encoding form
        if len(yTrue.shape) == 1:
            correct_confidences = yPredClipped[range(samples), yTrue]
        elif len(yTrue.shape) == 2:
            correct_confidences = np.sum(yPredClipped * yTrue, axis=1)

        negativeloglikelihoods = -np.log(correct_confidences)
        return negativeloglikelihoods

    # dvalues is softmax outputs
    def backward(self, dvalues, yTrue):
        samples = len(dvalues)
        # number of predicted values
        labels = len(dvalues[0])
        if len(yTrue.shape) == 1:
            # one-hot encoding
            yTrue = np.eye(labels)[yTrue]
        # calc change in loss w.r.t inputs
        self.dinputs = -yTrue / dvalues
        # this means when summing the gradients, the mean is calculated
        self.dinputs = self.dinputs / samples


# When backpropagating, either use the backward methods of the softmax and loss classes or used the combined class below which is faster
class SoftmaxLossCategoricalCrossEntropy(Loss):
    def __init__(self):
        self.softmax = SoftMax()
        self.loss = LossCategoricalCrossEntropy()

    def forward(self, inputs, yTrue):
        self.softmax.forward(inputs)
        self.output = self.softmax.output
        batch_loss = self.loss.calculate(self.output, yTrue)
        return batch_loss

    # the change in the loss function with respect to z is the predicted out of the softmax function minus the real output
    # dvalues is the softmax output
    def backward(self, dvalues, yTrue):
        samples = len(dvalues)
        if len(yTrue.shape) == 2:
            # scalar encoding
            yTrue = np.argmax(yTrue, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), yTrue] -= 1
        # mean loss
        self.dinputs = self.dinputs / samples


# Stochastic gradient descent, with a decay learning rate of 1/t
# uses adaptive momentum including RMS propagation
class Optimiser:
    def __init__(self, learning_rate=0.001, decay=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        # initial learning rate
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iterations = 0

    def pre_update_params(self):
        # if layer does not contain momentum/cache arrays, create them, filled with 0s
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)

    def update_params(self, layer):
        if not hasattr(layer, 'bias_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.bias)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        # update momentum with gradient, using RMS propagation equation
        layer.weight_momentums = layer.weight_momentums * self.beta_1 + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = layer.weight_momentums * self.beta_1 + (1 - self.beta_1) * layer.dbiases
        layer.weight_cache = layer.weight_cache * self.beta_2 + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = layer.bias_cache * self.beta_2 + (1 - self.beta_2) * layer.dbiases ** 2
        # get the corrected momentums/caches, with the 1/(1-beta^steps) correction
        # this gives a large starting momentum and cache but then approaches 1
        # self.iterations is 0 at first pass
        weight_momentums_corrected = layer.weight_momentums / (1 - (self.beta_1 ** (self.iterations + 1)))
        bias_momentums_corrected = layer.bias_momentums / (1 - (self.beta_1 ** (self.iterations + 1)))
        weight_cache_corrected = layer.weight_cache / (1 - (self.beta_2 ** (self.iterations + 1)))
        bias_cache_corrected = layer.bias_cache / (1 - (self.beta_2 ** (self.iterations + 1)))

        weights_update = -self.current_learning_rate * weight_momentums_corrected / (
                np.sqrt(weight_cache_corrected) + self.epsilon)
        biases_update = -self.current_learning_rate * bias_momentums_corrected / (
                np.sqrt(bias_cache_corrected) + self.epsilon)

        layer.weights += weights_update
        layer.biases += biases_update

    def post_update_params(self):
        self.iterations += 1


# n_inputs is the number of inputs from the input layer (number of features)
# n_neurons can be whatever you want
layer1 = Layer(2, 64, 5e-4, 5e-4)
activation1 = ReLU()

layer2 = Layer(64, 3)
activation2 = Softmax()

lossFunction = LossCategoricalCrossEntropy()
lossSoftmax = SoftmaxLossCategoricalCrossEntropy()

optimiser = Optimiser(learning_rate=0.05, decay=1e-7)


# Spiral Dataset
def generateData(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


# a is the learning rate
def forward(inputs):
    layer1.forward(inputs)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    return activation2.output


def main():
    # 100 size-2 feature sets, 3 times for 3 labels
    X, y = generateData(100, 3)
    delta = 0.025
    x = y = np.arange(-1.0, 1.0, delta)
    n = len(x)
    X, Y = np.meshgrid(x, y)
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Y = Y[::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")

    lossArray = []
    stepsArray = []
    for epoch in range(10001):
        layer1.forward(X)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        data_loss = lossSoftmax.calculate(layer2.output, y)
        regularisation_loss = lossSoftmax.regularization_loss(layer1) + lossSoftmax.regularization_loss(layer2)
        loss = data_loss + regularisation_loss
        predictions = np.argmax(lossSoftmax.output, axis=1)
        accuracy = np.mean(predictions == y)

        if epoch % 500:
            lossArray.append(loss)
            stepsArray.append(epoch)
        # input the softmax outputs and true y values
        lossSoftmax.backward(lossSoftmax.output, y)
        layer2.backward(lossSoftmax.dinputs)
        activation1.backward(layer2.dinputs)
        layer1.backward(activation1.dinputs)
        if epoch % 2000 == 0:
            print("epoch:", epoch)
            print("loss:", loss)
            print("data_loss:", data_loss)
            print("reg_loss:", regularisation_loss)
            print("acc:", accuracy)
            print("lr:", optimiser.current_learning_rate)
            print("-------------------------------------")

        optimiser.pre_update_params()
        optimiser.update_params(layer=layer1)
        optimiser.update_params(layer=layer2)
        optimiser.post_update_params()

    Z = forward(np.array(list(zip(X, Y))))
    Z = np.argmax(Z, axis=1).reshape(n, n)
    im = ax1.imshow(Z_, interpolation="bilinear", cmap="brg", extent=[-1, 1, -1, 1], alpha=0.5)
    ax1.axis('scaled')
    ax2.plot(stepsArray, lossArray)
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Loss")
    plt.show()
    plt.colorbar()


main()
