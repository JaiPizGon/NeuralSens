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
  if (is.function(type)) {
    # Custom function
    return(
      function(x) {
        eval(parse(text = paste0(deparse(type), collapse = ""),
                   keep.source = FALSE), envir = environment(type))(x)
      }
    )
  } else {
    # Switch case to define which function it returns
    switch(type,
           sigmoid = {
             return(
               function(x){
                 apply(x,c(1,2),
                       stats::plogis)
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
           # PReLU = {
           #   return(
           #     function(x,a){
           #       apply(x,c(1,2),
           #             function(y) {ifelse(y >= 0, y, a*y)})
           #     })
           # },
           # ELU = {
           #   return(
           #     function(x,a){
           #       apply(x,c(1,2),
           #             function(y) {ifelse(y >= 0, y, a*(exp(y)-1))})
           #     })
           # },
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
           },
           return(
             function(x){
               apply(x,c(1,2),type)
             }
           )
    )
  }

}

#' Derivative of activation function of neuron
#'
#' @description Evaluate derivative of activation function of a neuron
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
  if (is.function(type)) {
      # Custom function
      return(
        function(x) {
                eval(parse(text = paste0(deparse(type), collapse = ""),
                           keep.source = FALSE), envir = environment(type))(x)
        }
      )
  } else {
    # Switch case to define which value it returns
    switch(type,
           sigmoid = {
             return(function(x){
               if (length(x) == 1) {
                 (1 / (1 + exp(-x))) * (1 - 1 / (1 + exp(-x)))
               } else {
                 diag((1 / (1 + exp(-x))) * (1 - 1 / (1 + exp(-x))))
               }
             })
           },
           tanh = {
             return(function(x){
               if (length(x) == 1) {
                 1 - tanh(x)^2
               } else {
                 diag(1 - tanh(x)^2)
               }
             })
           },
           linear = {
             return(function(x){diag(length(x))})
           },
           ReLU = {
             return(function(x){
               if (length(x) == 1) {
                 ifelse(x >= 0, 1, 0)
               } else {
                 diag(ifelse(x >= 0, 1, 0))
               }
             })
           },
           # PReLU = {
           #   return(function(x,a){
           #     if (length(x) == 1) {
           #       ifelse(x >= 0, 1, a)
           #     } else {
           #       diag(ifelse(x >= 0, 1, a))
           #     }
           #   })
           # },
           # ELU = {
           #   return(function(x,a){
           #     if (length(x) == 1) {
           #       ifelse(x >= 0, 1,  a*(exp(x)-1) + a)
           #     } else {
           #       diag(ifelse(x >= 0, 1,  a*(exp(x)-1) + a))
           #     }
           #   })
           # },
           step = {
             return(function(x){
               if (length(x) == 1) {
                 ifelse(x != 0, 0, NA)
               } else {
                 diag(ifelse(x != 0, 0, NA))
               }
             })
           },
           arctan = {
             return(function(x){
               if (length(x) == 1) {
                 1/(x^2 + 1)
               } else {
                 diag(1/(x^2 + 1))
               }
             })
           },
           softPlus = {
             return(function(x){
               if (length(x) == 1) {
                 1/(1 + exp(-x))
               } else {
                 diag(1/(1 + exp(-x)))
               }
             })
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
}
#' Second derivative of activation function of neuron
#'
#' @description Evaluate second derivative of activation function of a neuron
#' @param type \code{character} name of the activation function
#' @param ... extra arguments needed to calculate the functions
#' @return \code{numeric} output of the neuron
#' @examples
#' # Return derivative of the sigmoid activation function of a neuron
#' ActivationFunction <- Der2ActFunc("sigmoid")
#' # Return derivative of the tanh activation function of a neuron
#' ActivationFunction <- Der2ActFunc("tanh")
#' # Return derivative of the activation function of several layers of neurons
#' actfuncs <- c("linear","sigmoid","linear")
#' ActivationFunctions <- sapply(actfuncs, Der2ActFunc)
#' @export Der2ActFunc
Der2ActFunc <- function(type = "sigmoid", ...) {
  if (is.function(type)) {
      # Custom function
      return(
        function(x) {
                eval(parse(text = paste0(deparse(type), collapse = ""),
                           keep.source = FALSE), envir = environment(type))(x)
        }
      )
  } else {
    # Switch case to define which value it returns
    switch(type,
           sigmoid = {
             return(function(x){
               if (length(x) == 1) {
                 y <-  1/(1 + exp(-x)) * (1 - 1/(1 + exp(-x))) * (1 - 2/(1 + exp(-x)))
               } else {
                 NeuralSens::diag3Darray(1/(1 + exp(-x)) * (1 - 1/(1 + exp(-x))) * (1 - 2/(1 + exp(-x))))
               }
             })
           },
           tanh = {
             return(function(x){
               if (length(x) == 1) {
                 -2 * tanh(x) * (1 - tanh(x)^2)
               } else {
                 NeuralSens::diag3Darray(-2 * tanh(x) * (1 - tanh(x)^2))
               }
             })
           },
           linear = {
             return(function(x){array(0, dim = rep(length(x),3))})
           },
           ReLU = {
             return(function(x){array(0, dim = rep(length(x),3))})
           },
           # PReLU = {
           #   return(function(x,a){
           #     if (length(x) == 1) {
           #       ifelse(x >= 0, 1, a)
           #     } else {
           #       diag(ifelse(x >= 0, 1, a))
           #     }
           #   })
           # },
           # ELU = {
           #   return(function(x,a){
           #     if (length(x) == 1) {
           #       ifelse(x >= 0, 1,  a*(exp(x)-1) + a)
           #     } else {
           #       diag(ifelse(x >= 0, 1,  a*(exp(x)-1) + a))
           #     }
           #   })
           # },
           step = {
             return(function(x){
               if (length(x) == 1) {
                 ifelse(x != 0, 0, NA)
               } else {
                 NeuralSens::diag3Darray(ifelse(x != 0, 0, NA))
               }
             })
           },
           arctan = {
             return(function(x){
               if (length(x) == 1) {
                 -2 * x / ((1 + x^2)^2)
               } else {
                 NeuralSens::diag3Darray(-2 * x / ((1 + x^2)^2))
               }
             })
           },
           softPlus = {
             return(function(x){
               if (length(x) == 1) {
                 exp(-x)/((1 + exp(-x))^2)
               } else {
                 NeuralSens::diag3Darray(exp(-x)/((1 + exp(-x))^2))
               }
             })
           },
           softmax = {
             return(
               function(x) {
                 x <- exp(x - max(x)) / sum(exp(x - max(x))) #Numerical stability
                 # build 'delta' arrays
                 d_i_m <- array(diag(length(x)), dim = rep(length(x), 3))
                 d_i_p <- aperm(d_i_m, c(3,2,1))
                 d_m_p <- aperm(d_i_m, c(2,3,1))
                 # Build 'a' arrays
                 ai <- array(x %*% t(rep(1, length(x))), dim = rep(length(x), 3))
                 am <- aperm(ai, c(2,1,3))
                 ap <- aperm(ai, c(2,3,1))
                 # Create second derivative array
                 x <- ai * ((d_i_p - ap) * (d_i_m - am) - am * (d_m_p * ap))
                 return(x)
               }
             )
           }
    )
  }
}

#' Loss function
#'
#' @description Evaluate loss function of a neuron
#' @param type \code{character} name of the loss function
#' @param ... extra arguments needed to calculate the functions
#' @return \code{list} containing the \code{numeric} error of the neural network
#' and a \code{numeric} vector to codify if it is a regression loss (0), a
#' binary classification loss (1) or a multiclass classification loss (2)
#' @examples
#' lossFunction <- LossFunc("RMSE")
#' @export LossFunc
LossFunc <- function(type = "RMSE", ...) {
  if (is.list(type)) {
    # Custom function
    return(
      list(
        function(y, y_pred) {
          eval(parse(text = paste0(deparse(type[[1]]), collapse = ""),
                     keep.source = FALSE),
               envir = environment(type[[1]]))(y, y_pred)
        }, type[[2]])
    )
  } else {
    switch(type,
           ####### REGRESSION ERROR
           RMSE = {
             return(
               list(
                 function(y, y_pred) {
                  error <- sqrt(mean((y - y_pred)^2))
                  return(error)
                 }, 0)
             )
           },
           R2 = {
             return(
               list(
                 function(y, y_pred) {
                  error <- 1 - sum((y - y_pred)^2) / sum((mean(y) - y_pred)^2)
                  return(error)
                 }, 0)
             )
           },
           MBE = {
             return(
               list(
                 function(y, y_pred) {
                  error <- mean(y - y_pred)
                  return(error)
                }, 0)
             )
           },
           MAE = {
             return(
               list(
                 function(y, y_pred) {
                  error <- mean(abs(y - y_pred))
                  return(error)
                }, 0)
             )
           },
           MSE = {
             return(
               list(
                 function(y, y_pred) {
                  error <- mean((y - y_pred)^2)
                  return(error)
               }, 0)
             )
           },
           SSE = {
             return(
               list(
                 function(y, y_pred) {
                   error <- sum((y - y_pred)^2)
                   return(error)
                 }, 0)
             )
           },
           MSLE = {
             return(
               list(
                 function(y, y_pred) {
                  error <- mean((log10(y + 1) - log10(y_pred + 1))^2)
                  return(error)
                }, 0)
             )
           },
           MAPE = {
             return(
               list(
                 function(y, y_pred) {
                  error <- mean(abs(y - y_pred) / y * 100)
                  return(error)
                }, 0)
             )
           },
           Huber = {
             return(
               list(
                 function(y, y_pred, delta) {
                  dif_y_y_pred <- abs(y - y_pred)
                  error <- sum(ifelse(dif_y_y_pred < delta,
                                     (y - y_pred) ^ 2,
                                     2 * delta * dif_y_y_pred - delta ^ 2))
                  return(error)
               }, 0)
             )
           },
           logcosh = {
             return(
               list(
                 function(y, y_pred) {
                  error <- sum(log10(cosh(y_pred - y)))
                  return(error)
                }, 0)
             )
           },
           ######### CLASSIFICATION ERROR
           accuracy = {
             return(
               list(
                 function(y, y_pred) {
                   error <- mean(y == y_pred)
                 }, c(1, 2)
               )
             )
           },
           crossentropy = {
             return(
               list(
                 function(y, y_pred) {
                    error <- -mean(y * log(y_pred))
                    return(error)
                 }, c(1, 2))
             )
           }
           # Hinge = {
           #   return(
           #     list(
           #       function(y, y_pred) {
           #         hinge_err <- y - (1 - 2 * y) * y_pred
           #         error <- sum(ifelse(hinge_err < 0, 0, hinge_err))
           #         return(error)
           #       }, c(1))
           #   )
           # },
           # sqHinge = {
           #   return(
           #     list(
           #       function(y, y_pred) {
           #         hinge_err <- y - (1 - 2 * y) * y_pred
           #         error <- sum(ifelse(hinge_err < 0, 0, hinge_err)^2)
           #         return(error)
           #       }, c(1))
           #   )
           # },
           # KL = {
           #   return(
           #     list(
           #       function(y, y_pred) {
           #         error <- sum(y_pred * log(y_pred / y))
           #         return(error)
           #       }, 2)
           #   )
           # }
           )
  }
}
#' Derivative of loss function
#'
#' @description Evaluate derivative of loss function of a neuron
#' @param type \code{character} name of the loss function
#' @param ... extra arguments needed to calculate the functions
#' @return \code{numeric} output of the neuron
#' @examples
#' lossFunction <- DerLossFunc("RMSE")
#' @export DerLossFunc
DerLossFunc <- function(type = "RMSE", ...) {
  if (is.list(type)) {
    # Custom function
    return(
        function(y, y_pred) {
          eval(parse(text = paste0(deparse(type[[1]]), collapse = ""),
                     keep.source = FALSE),
               envir = environment(type[[1]]))(y, y_pred)
        }
    )
  } else {
    switch(type,
           RMSE = {
             return(
               function(y, y_pred) {
                 N <- length(y)
                 dif <- y - y_pred
                 der <- sqrt(N)/N * dif / sqrt(sum(dif^2))
                 return(der)
               }
             )
           },
           R2 = {
             return(
               function(y, y_pred) {
                 SSM = (y_pred - mean(y))^2
                 derSSM = 2 * (y_pred - mean(y))
                 SSE = (y - y_pred)^2
                 derSSE = -2 * (y - y_pred)
                 der = - (derSSE * SSM + SSE * derSSM) / SSM ^ 2
                 return(der)
               }
             )
           },
           MBE = {
             return(
                 function(y, y_pred) {
                   der <- -1
                   return(der)
                 }
             )
           },
           MAE = {
             return(
                 function(y, y_pred) {
                   der <- (y_pred - y) / abs(y - y_pred)
                   return(der)
                 }
             )
           },
           MSE = {
             return(
                 function(y, y_pred) {
                   der <- 2 *(y_pred - y)
                   return(der)
                 }
             )
           },
           SSE = {
             return(
                 function(y, y_pred) {
                   der <- 2 *(y_pred - y)
                   return(der)
                 }
             )
           },
           MSLE = {
             return(
                 function(y, y_pred) {
                   der <- 2 * (log(y_pred+1) - log(y+1)) / ((y_pred+1) * log(10)^2)
                   return(der)
                 }
             )
           },
           MAPE = {
             return(
                 function(y, y_pred) {
                   der <- (y_pred - y) / (y * abs(y - y_pred))
                   return(der)
                 }
             )
           },
           Huber = {
             return(
                 function(y, y_pred, delta) {
                   dif_y_y_pred <- abs(y - y_pred)
                   der <- sum(ifelse(dif_y_y_pred < delta,
                                    -2 * (y - y_pred),
                                    2 * delta * (y_pred - y) / dif_y_y_pred))
                   return(der)
                 }
             )
           },
           logcosh = {
             return(
                 function(y, y_pred) {
                   error <- tanh(y_pred - y) / log(10)
                   return(error)
                 }
             )
           },
           ######### CLASSIFICATION ERROR
           crossentropy = {
             return(
               function(y, y_pred) {
                 der <- -y / y_pred + (1 - y) / (1 - y_pred)
                 return(der)
               }
             )
           }
           )
  }
}
