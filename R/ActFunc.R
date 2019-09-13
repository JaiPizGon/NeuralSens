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
#' actfuncs <- c("linear","sigmoid","linear")
#' ActivationFunctions <- sapply(actfuncs, ActFunc)
#' @export ActFunc
ActFunc <- function(type = "sigmoid", ...) {
  # Switch case to define which function it returns
  switch(type,
         sigmoid = {
           return(
             function(x){
               apply(x,c(1,2),
                     function(y) {1 / (1 + exp(-y))})
               })
         },
         tanh = {
           return(
             function(x){
               apply(x,c(1,2),
                     function(y) {tanh(y)})
               })
         },
         linear = {
           return(
             function(x){
               apply(x,c(1,2),
                     function(y) {y})
               })
         },
         ReLU = {
           return(
             function(x){
               apply(x,c(1,2),
                     function(y) {max(0,y)})
               })
         },
         PReLU = {
           return(
             function(x,a){
               apply(x,c(1,2),
                     function(y) {ifelse(y >= 0, y, a*y)})
           })
         },
         ELU = {
           return(
             function(x,a){
               apply(x,c(1,2),
                     function(y) {ifelse(y >= 0, y, a*(exp(y)-1))})
           })
         },
         step = {
           return(
             function(x){
               apply(x,c(1,2),
                     function(y) {ifelse(y >= 0, 1, 0)})
           })
         },
         arctan = {
           return(
             function(x){
               apply(x,c(1,2),
                     function(y) {atan(y)})
               })
         },
         softPlus = {
           return(
             function(x){
               apply(x,c(1,2),
                     function(y) {log(1 + exp(y))})
               })
         },
         softmax = {
           return(
             function(x) {
               for(i in 1:nrow(x)) {
                 x[i,] <- exp(x[i,] - max(x[i,])) /  #Numerical stability
                   sum(exp(x[i,] - max(x[i,])))
               }
               return(x)
             }
           )
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
#' actfuncs <- c("linear","sigmoid","linear")
#' ActivationFunctions <- sapply(actfuncs, DerActFunc)
#' @export DerActFunc
DerActFunc <- function(type = "sigmoid", ...) {
  # Switch case to define which value it returns
  switch(type,
         sigmoid = {
           return(function(x){
             diag((1 / (1 + exp(-x))) *
                    (1 - 1 / (1 + exp(-x))))
             })
         },
         tanh = {
           return(function(x){diag(1 - tanh(x)^2)})
         },
         linear = {
           return(function(x){diag(length(x))})
         },
         ReLU = {
           return(function(x){
             diag(ifelse(x >= 0, 1, 0))
           })
         },
         PReLU = {
           return(function(x,a){
             diag(ifelse(x >= 0, 1, a))
           })
         },
         ELU = {
           return(function(x,a){
             diag(ifelse(x >= 0, 1, a*(exp(x)-1) + a))
           })
         },
         step = {
           return(function(x){
             diag(ifelse(x != 0, 0, NA))
           })
         },
         arctan = {
           return(function(x){diag(1/(x^2 + 1))})
         },
         softPlus = {
           return(function(x){diag(1/(1 + exp(-x)))})
         },
         softmax = {
           return(
             function(x) {
               x <- exp(x - max(x)) /  #Numerical stability
                 sum(exp(x - max(x)))
               # Derivative as in http://saitcelebi.com/tut/output/part2.html
               x <- x %*% t(rep(1,length(x))) * (diag(length(x)) - rep(1,length(x)) %*% t(x))

               return(x)
             }
           )
         }
  )
}
