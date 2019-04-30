#' Activation function of neuron
#'
#' @description Evaluate activation function of a neuron
#' @param type \code{character} name of the activation function
#' @param ... extra arguments needed to calculate the functions
#' @return \code{numeric} output of the neuron
#' @examples
#' # Return the sigmoid activation function of a neuron
#' ActivationFunction <- ActFunc("sigmoid")
#' # Return the tanh activation function of a neuron
#' ActivationFunction <- ActFunc("tanh")
#' # Return the activation function of several layers of neurons
#' actfuncs <- c("linear","sigmoid","linear)
#' ActivationFunctions <- sapply(actfuncs, ActFunc)
#' @export ActFunc
ActFunc <- function(type = "sigmoid", ...) {
  # Switch case to define which function it returns
  switch(type,
         sigmoid = {
           return(function(x){1 / (1 + exp(-x))})
         },
         tanh = {
           return(function(x){tanh(x)})
         },
         linear = {
           return(function(x){x})
         },
         ReLU = {
           return(function(x){max(0,x)})
         },
         PReLU = {
           return(function(x,a) {
             ifelse(x >= 0, x, a*x)
           })
         },
         ELU = {
           return(function(x,a) {
             ifelse(x >= 0, x, a*(exp(x)-1))
           })
         },
         step = {
           return(function(x) {
             ifelse(x >= 0, 1, 0)
           })
         },
         arctan = {
           return(function(x){atan(x)})
         },
         softPlus = {
           return(function(x){log(1 + exp(x))})
         }
         )
}

#' Derivate activation function of neuron
#'
#' @description Evaluate derivate of activation function of a neuron
#' @param type \code{character} name of the activation function
#' @param ... extra arguments needed to calculate the functions
#' @return \code{numeric} output of the neuron
#' @examples
#' # Return derivative of the sigmoid activation function of a neuron
#' ActivationFunction <- DerActFunc("sigmoid")
#' # Return derivative of the tanh activation function of a neuron
#' ActivationFunction <- DerActFunc("tanh")
#' # Return derivative of the activation function of several layers of neurons
#' actfuncs <- c("linear","sigmoid","linear)
#' ActivationFunctions <- sapply(actfuncs, DerActFunc)
#' @export DerActFunc
DerActFunc <- function(type = "sigmoid", ...) {
  # Switch case to define which value it returns
  switch(type,
         sigmoid = {
           return(function(x){(1 / (1 + exp(-x))) *
                    (1 - 1 / (1 + exp(-x)))})
         },
         tanh = {
           return(function(x){1 - tanh(x)^2})
         },
         linear = {
           return(function(x){1})
         },
         ReLU = {
           return(function(x){
             ifelse(x >= 0, 1, 0)
           })
         },
         PReLU = {
           return(function(x,a){
             ifelse(x >= 0, 1, a)
           })
         },
         ELU = {
           return(function(x,a){
             ifelse(x >= 0, 1, a*(exp(x)-1) + a)
           })
         },
         step = {
           return(function(x){
             ifelse(x != 0, 0, NA)
           })
         },
         arctan = {
           return(function(x){1/(x^2 + 1)})
         },
         softPlus = {
           return(function(x){1/(1 + exp(-x))})
         }
  )
}
