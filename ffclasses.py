import numpy as np
import mnist as mn

class SigmoidLayer: 
    
    def __init__ (self, input_len, nodes):
        self.weights = np.random.randn(input_len,nodes)/input_len
        self.biases = np.random.randn(nodes) 
    
    def forward(self, input):
        self.last_input_shape = input.shape
        self.last_input = input
        total = np.dot(input, self.weights) + self.biases
        self.last_total = total

        exp = np.exp(-total)
        return 1/(1+exp)

    def backward(self, dLossdOutput, learnrate):
        exp = np.exp(-self.last_total)
        dOutputdTotal = exp/ (1+exp) **2
        dLossdTotal = dOutputdTotal*dLossdOutput
        dTotalsdWeights = self.last_input
        dTotalsdBiases = 1
        dTotalsdInputs = self.weights
        dLossdWeights = dTotalsdWeights[np.newaxis].T @ dLossdTotal[np.newaxis]
        dLossdBiases = dLossdTotal*dTotalsdBiases
        dLossdInputs = dTotalsdInputs@dLossdTotal
        
        self.weights -= learnrate*dLossdWeights
        self.biases -=  learnrate*dLossdBiases
        return dLossdInputs



class SoftmaxLayer:

    def __init__ (self, input_len, nodes):
        self.weights = np.random.randn(input_len,nodes)/input_len
        self.biases = np.random.randn(nodes) 

    def forward(self, input):
        #performs a forward pass of the softmax layer. Pretty self explanatory except for caching values
        #we will use those values in the backprop stage which is why theyre cached
        self.last_input_shape = input.shape
        self.last_input = input
        total = np.dot(input, self.weights) + self.biases
        self.last_total = total
        exp = np.exp(total)
        return exp/np.sum(exp, axis=0)

    def backward(self, dLossdOutput,learnrate):
        #performs back propagation phase for the softmax layer
        #takes in the gradient of loss with respect to the output from this layer
        #outputs the gradient of loss with respect to the *input* from this layer
        #there's gonna be a Lot of comments here bc Theo is bad at calculus ;_;


        for i, gradient in enumerate(dLossdOutput):
            #this for loop finds the element of dLossdOutput that is non-zero, and uses that element to 
            #figure out the gradient of output with respect to totals
            if gradient == 0:
                continue
            
            #this gets us back e^totals
            total_exp = np.exp(self.last_total)

            #this gets us back the sum of e^totals
            total_exp_sum = np.sum(total_exp)

            #ok so here's where lots of comments happen haha
            #we're looking for the gradient of output with respect to the totals 
            #BUT
            #we're only looking for the gradient of THIS ONE ELEMENT OF THE OUTPUT
            #the element that corresponds to the correct label, ie the number that we want the network to pick
            #and this is why we get everything set up using the for loop so that we're only accessing that element
            #of the totals and of the output.
            #total_exp[i] is the element of the total that corresponds to the choice that we want before we feed it into softmax
            #that's why it's special
            #i'm writing this down bc this math is hard and i hate it and hopefully this helps
            #pretty overcaffeinated rn

            #this is the gradient of output[i] with respect to each element of the total (pre-softmax) except for total[i]
            #because if its a different element of the total then 
            # we have to differentiate total_exp[i]/the sum(including total_exp[j])
            #which is d(output[i])/d(total_exp_sum) *d(total_exp_sum)/d(total_exp[j])
            #which comes out to... -total_exp[i]*total_exp[j]/ total_exp_sum^2

            dOutputdTotals = -total_exp[i] *total_exp/total_exp_sum **2

            #if it's the same element of the totals then its different
            #then we're differentiating total_exp[i]/total_exp_sum (that whole thing is output[i]) with respect to total_exp[i]
            #which you can't break down using chain rule since total_exp[i] is in the numerator 
            # so its just d(output[i])/d(total_exp[i])
            #which breaks down to:

            dOutputdTotals[i] = total_exp[i] * (total_exp_sum-total_exp[i])/(total_exp_sum**2)
            if gradient !=0:
                dLossdTotals = gradient * dOutputdTotals


            #now that we have the gradiens of output wrt totals we can start to put it all together
            dTotalsdWeights = self.last_input
            dTotalsdBiases = 1
            dTotalsdInputs = self.weights


            dLossdWeights = dTotalsdWeights[np.newaxis].T @ dLossdTotals[np.newaxis]
            #the previous line is gonna need some explanation
            #what we want is a 2d array the same size as self.weights 
            #the newaxis commands turn these into matrices with an extra dimension of length 1 instead of arrays
            #and then the .T transposes dTotalsdWeights so that it can be multiplied by dLossdTotals
            #this way we get a matrix with dimensions of input_length, output_length, which is the same size as self.weights
            dLossdBiases = dLossdTotals*dTotalsdBiases

            dLossdInputs = dTotalsdInputs@dLossdTotals
            #above we multiplied a matrix with size inputlength, nodes and a matrix with size nodes, 1 to get a matrix with 
            #size inputlength, 1
            #which is what we want : )

            self.weights -= learnrate*dLossdWeights
            self.biases -=  learnrate*dLossdBiases
            return dLossdInputs

        

