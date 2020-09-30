import numpy as np
import mnist as mn
from ffclasses import SigmoidLayer, SoftmaxLayer

training_images = mn.test_images()

training_labels = mn.test_labels()


sigmoid = SigmoidLayer(784, 200)

sigmoid2 = SigmoidLayer(200, 80)

softmax = SoftmaxLayer(80, 10)


def forward(image, label):
    #completes a forward pass of the two layers in the neural network

    #first we flatten the 28*28 array into a single array bc it's not convolutional and idc about the shape rn
    out = image.flatten()

    #then we normalize the input by changing it from 0-255 to -0.5 -> 0.5
    out = sigmoid.forward((out/255)-0.5)
    out = sigmoid2.forward(out)
    out = softmax.forward(out)

    loss = -np.log(out[label])
    
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

def train(image, label, lr = 0.005):
    output, loss, acc = forward(image, label)

    gradient = np.zeros(10)
    #the gradient of loss for each label except the correct one will be zero
    gradient[label] = -1/output[label]
    #this is the gradient of the cross-entropy loss with respect to the output of softmax

    gradient = softmax.backward(gradient, lr)
    #todo do the backward step for the sigmoid layer
    gradient = sigmoid2.backward(gradient, lr)
    gradient = sigmoid.backward(gradient, lr)

    return loss, acc



total_loss = 0
number_correct = 0

while (True):

    for i, (im,label) in enumerate(zip(training_images, training_labels)):
        #enumerate makes it easier to do a for loop basically
        # lets you loop over something with an automatic counter
        #zip returns an iterator of tuples, essentially returning a tuple of training_images[i], training_labels[i]


        
        if i % 100 == 99:
            print(
            '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
            (i + 1, loss / 100, number_correct)
            )
            total_loss = 0
            number_correct = 0

        loss, acc = train(im, label)

        total_loss+=loss
        number_correct += acc





        