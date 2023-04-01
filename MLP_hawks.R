#install.packages("caret")

library(caret)

# select 70% of data for training. 
trainrows <- sample(1:nrow(hawksOmit), replace = F, size = nrow(hawksOmit)*0.70) 

X = hawksOmit[trainrows,1:4] #input features (data)
Y = hawksOmit[trainrows, 5] #target variables (label)
head(X)
nrow(X)
table(Y)

#sigmoid activation function
sigmoid<-function(x) return (1/(1+exp(-x)))

#sigmoid function derivative for backpropagation
sigmoid_derivative<-function(x) return (x*(1-x))

#attempting reLU activation function
#relu <- function(x) {
#  (max(0, x))
#}

#relu derivative for backpropagation
#relu_derivative <- function(x) {
#  (ifelse(x > 0, 1, 0))
#}

# variable initialization
learning_rate=0.001
epochs=500
inputlayer_neurons=ncol(X)
hiddenlayer_neurons=15 #by increasing the hidden layers the model becomes 'multi-layer'
outputlayer_neurons=1

#initializing model parameters (random weights and bias)
weight=matrix(rnorm(inputlayer_neurons*hiddenlayer_neurons,mean=0,sd=1), inputlayer_neurons, hiddenlayer_neurons)
bias_in=runif(hiddenlayer_neurons)
bias_in_temp=rep(bias_in, nrow(X))
bias_hidden=matrix(bias_in_temp, nrow = nrow(X), byrow = FALSE)
weights_between_layers=matrix(rnorm(hiddenlayer_neurons*outputlayer_neurons,mean=0,sd=1), hiddenlayer_neurons, outputlayer_neurons)

bias_out=runif(outputlayer_neurons)
bias_out_temp=rep(bias_out,nrow(X))
bias_output=matrix(bias_out_temp,nrow = nrow(X),byrow = FALSE)


#convert to matrix for calculating
X<- as.matrix(X)

Y<-as.matrix(Y)

# forward propagation function
for(i in 1:epochs){
  
  #sigmoid activation
  hiddenlayer_input1=X%*%weight
  hiddenlayer_input=hiddenlayer_input1+bias_hidden
  hiddenlayer_activations=sigmoid(hiddenlayer_input)
  outputlayer_input1=hiddenlayer_activations%*%weights_between_layers
  outputlayer_input=outputlayer_input1+bias_output
  output = sigmoid(outputlayer_input)
  
  #attempt at relu activation
  #hiddenlayer_input1=X%*%weight
  #hiddenlayer_input=hiddenlayer_input1+bias_hidden
  #hiddenlayer_activations=relu(hiddenlayer_input)
  #outputlayer_input1=hiddenlayer_activations%*%weights_between_layers
  #outputlayer_input=outputlayer_input1+bias_output
  #output = relu(outputlayer_input)
  
  # Back Propagation function
  
  Error=Y-output
  
  #sigmoid backprop
  slope_output_layer=sigmoid_derivative(output)
  slope_hidden_layer=sigmoid_derivative(hiddenlayer_activations)
  delta_output=Error*slope_output_layer
  
  #attempt at relu backprop
  #slope_output_layer=relu_derivative(output)
  #slope_hidden_layer=relu_derivative(hiddenlayer_activations)
  #delta_output=E*slope_output_layer
  
  hiddenlayer_error=delta_output%*%t(weights_between_layers)
  
  delta_hiddenlayer=hiddenlayer_error*slope_hidden_layer
  weights_between_layers= weights_between_layers + (t(hiddenlayer_activations)%*%delta_output)*learning_rate
  bias_output= bias_output+rowSums(delta_output)*learning_rate
  weight = weight +(t(X)%*%delta_hiddenlayer)*learning_rate
  bias_hidden = bias_hidden + rowSums(delta_hiddenlayer)*learning_rate
  
}

output

#Attempting cross validation on MLP
cat("
hawksOmit[sapply(hawksOmit, is.factor)] <- data.matrix(hawksOmit[sapply(hawksOmit, is.factor)])
results <- train(
  validationX = hawksOmit[,1:4],       # input features
  validationY = hawksOmit[,5],         # target variable
  method = mlp_model(output),  # gradient descent function
  trControl = trainControl(
    method = cv,        # Use cross validation
    number = 5            # Use 5 folds
  ),
  gridTuning = expand.grid(
    inputlayer_neurons = 4,   # input neurons
    hiddenlayer_neurons = 15, # hidden neurons
    outputlayer_neurons = 1,  # output neurons
    epochs = 500,             # number of epochs
    learning_rate = 0.1       # learning rate
  )
)
head(validationX)
head(validationY)")