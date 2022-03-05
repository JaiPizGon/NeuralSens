#' Constructor of the JerkMLP Class
#'
#' Create an object of JerkMLP class
#' @param sens \code{list} of sensitivity measures, one \code{list} per output neuron
#' @param raw_sens \code{list} of sensitivities, one \code{array} per output neuron
#' @param mlp_struct \code{numeric} vector describing the structur of the MLP model
#' @param trData \code{data.frame} with the data used to calculate the sensitivities
#' @param coefnames \code{character} vector with the name of the predictor(s)
#' @param output_name \code{character} vector with the name of the output(s)
#' @return \code{JerkMLP} object
#' @export JerkMLP
JerkMLP <- function(sens = list(),
                    raw_sens = list(),
                    mlp_struct = numeric(),
                    trData = data.frame(),
                    coefnames = character(),
                    output_name = character()
                    ) {
  stopifnot(is.list(sens))
  stopifnot(is.list(raw_sens))
  stopifnot(is.numeric(mlp_struct))
  stopifnot(is.character(coefnames))
  stopifnot(is.character(output_name))
  stopifnot(length(sens) == mlp_struct[length(mlp_struct)])
  stopifnot(length(raw_sens) == mlp_struct[length(mlp_struct)])
  stopifnot(length(output_name) == mlp_struct[length(mlp_struct)])

  structure(
    list(
      sens = sens,
      raw_sens = raw_sens,
      mlp_struct = mlp_struct,
      trData = trData,
      coefnames = coefnames,
      output_name = output_name
    ),
    class = "JerkMLP"
  )
}
#' Check if object is of class \code{JerkMLP}
#'
#' Check if object is of class \code{JerkMLP}
#' @param object \code{JerkMLP} object
#' @return \code{TRUE} if \code{object} is a \code{JerkMLP} object
#' @export is.JerkMLP
is.JerkMLP <- function(object) {
  any(class(object) == "JerkMLP")
}
#' Summary Method for the JerkMLP Class
#'
#' Print the sensitivity metrics of a \code{JerkMLP} object.
#' This metrics are the mean sensitivity, the standard deviation
#' of sensitivities and the mean of sensitivities square
#' @param object \code{JerkMLP} object created by \code{\link[NeuralSens]{JerkianMLP}}
#' @param ... additional parameters
#' @return summary object of the \code{JerkMLP} object passed
#' @examples
#' ## Load data -------------------------------------------------------------------
#' data("DAILY_DEMAND_TR")
#' fdata <- DAILY_DEMAND_TR
#'
#' ## Parameters of the NNET ------------------------------------------------------
#' hidden_neurons <- 5
#' iters <- 250
#' decay <- 0.1
#'
#' ################################################################################
#' #########################  REGRESSION NNET #####################################
#' ################################################################################
#' ## Regression dataframe --------------------------------------------------------
#' # Scale the data
#' fdata.Reg.tr <- fdata[,2:ncol(fdata)]
#' fdata.Reg.tr[,3] <- fdata.Reg.tr[,3]/10
#' fdata.Reg.tr[,1] <- fdata.Reg.tr[,1]/1000
#'
#' # Normalize the data for some models
#' preProc <- caret::preProcess(fdata.Reg.tr, method = c("center","scale"))
#' nntrData <- predict(preProc, fdata.Reg.tr)
#'
#' #' ## TRAIN nnet NNET --------------------------------------------------------
#' # Create a formula to train NNET
#' form <- paste(names(fdata.Reg.tr)[2:ncol(fdata.Reg.tr)], collapse = " + ")
#' form <- formula(paste(names(fdata.Reg.tr)[1], form, sep = " ~ "))
#'
#' set.seed(150)
#' nnetmod <- nnet::nnet(form,
#'                            data = nntrData,
#'                            linear.output = TRUE,
#'                            size = hidden_neurons,
#'                            decay = decay,
#'                            maxit = iters)
#' # Try JerkianMLP
#' sens <- NeuralSens::JerkianMLP(nnetmod, trData = nntrData, plot = FALSE)
#' summary(sens)
#' @method summary JerkMLP
#' @export
summary.JerkMLP <- function(object, ...) {
  class(object) <- c("summary.JerkMLP", class(object))
  print(object, ...)
}
#' Print method of the summary JerkMLP Class
#'
#' Print the sensitivity metrics of a \code{JerkMLP} object.
#' This metrics are the mean sensitivity, the standard deviation
#' of sensitivities and the mean of sensitivities square
#' @param x \code{summary.JerkMLP} object created by summary method of \code{JerkMLP} object
#' @param round_digits \code{integer} number of decimal places, default \code{NULL}
#' @param ... additional parameters
#' @examples
#' ## Load data -------------------------------------------------------------------
#' data("DAILY_DEMAND_TR")
#' fdata <- DAILY_DEMAND_TR
#'
#' ## Parameters of the NNET ------------------------------------------------------
#' hidden_neurons <- 5
#' iters <- 250
#' decay <- 0.1
#'
#' ################################################################################
#' #########################  REGRESSION NNET #####################################
#' ################################################################################
#' ## Regression dataframe --------------------------------------------------------
#' # Scale the data
#' fdata.Reg.tr <- fdata[,2:ncol(fdata)]
#' fdata.Reg.tr[,3] <- fdata.Reg.tr[,3]/10
#' fdata.Reg.tr[,1] <- fdata.Reg.tr[,1]/1000
#'
#' # Normalize the data for some models
#' preProc <- caret::preProcess(fdata.Reg.tr, method = c("center","scale"))
#' nntrData <- predict(preProc, fdata.Reg.tr)
#'
#' #' ## TRAIN nnet NNET --------------------------------------------------------
#' # Create a formula to train NNET
#' form <- paste(names(fdata.Reg.tr)[2:ncol(fdata.Reg.tr)], collapse = " + ")
#' form <- formula(paste(names(fdata.Reg.tr)[1], form, sep = " ~ "))
#'
#' set.seed(150)
#' nnetmod <- nnet::nnet(form,
#'                            data = nntrData,
#'                            linear.output = TRUE,
#'                            size = hidden_neurons,
#'                            decay = decay,
#'                            maxit = iters)
#' # Try JerkianMLP
#' sens <- NeuralSens::JerkianMLP(nnetmod, trData = nntrData, plot = FALSE)
#' print(summary(sens))
#' @method print summary.JerkMLP
#' @export
print.summary.JerkMLP <- function(x, round_digits = NULL, ...) {
  cat("Jerkian array of ", paste(x$mlp_struct, collapse = "-"), " MLP network.\n\n", sep = "")
  # cat("Measures are calculated using the partial derivatives of ", x$layer_end, " layer's ",
  #     ifelse(x$layer_end_input,"input","output"), "\nwith respect to ", x$layer_origin, " layer's ",
  #     ifelse(x$layer_origin_input,"input","output"),".\n\n",
  #     sep = "")
  if (!is.null(round_digits)) {
    if (round_digits >= 0) {
      x$sens <- lapply(x$sens,
                       function(y) {
                         lapply(y, function(z){
                           round(z, round_digits)
                         })
                       })
    }
  }
  cat("Jerkian measures of each output:\n")
  invisible(print(x$sens))
}

#' Print method for the JerkMLP Class
#'
#' Print the sensitivities of a \code{JerkMLP} object.
#' @param x \code{JerkMLP} object created by \code{\link[NeuralSens]{JerkianMLP}}
#' @param n \code{integer} specifying number of sensitivities to print per each output
#' @param round_digits \code{integer} number of decimal places, default \code{NULL}
#' @param ... additional parameters
#' @examples
#' ## Load data -------------------------------------------------------------------
#' data("DAILY_DEMAND_TR")
#' fdata <- DAILY_DEMAND_TR
#'
#' ## Parameters of the NNET ------------------------------------------------------
#' hidden_neurons <- 5
#' iters <- 250
#' decay <- 0.1
#'
#' ################################################################################
#' #########################  REGRESSION NNET #####################################
#' ################################################################################
#' ## Regression dataframe --------------------------------------------------------
#' # Scale the data
#' fdata.Reg.tr <- fdata[,2:ncol(fdata)]
#' fdata.Reg.tr[,3] <- fdata.Reg.tr[,3]/10
#' fdata.Reg.tr[,1] <- fdata.Reg.tr[,1]/1000
#'
#' # Normalize the data for some models
#' preProc <- caret::preProcess(fdata.Reg.tr, method = c("center","scale"))
#' nntrData <- predict(preProc, fdata.Reg.tr)
#'
#' #' ## TRAIN nnet NNET --------------------------------------------------------
#' # Create a formula to train NNET
#' form <- paste(names(fdata.Reg.tr)[2:ncol(fdata.Reg.tr)], collapse = " + ")
#' form <- formula(paste(names(fdata.Reg.tr)[1], form, sep = " ~ "))
#'
#' set.seed(150)
#' nnetmod <- nnet::nnet(form,
#'                            data = nntrData,
#'                            linear.output = TRUE,
#'                            size = hidden_neurons,
#'                            decay = decay,
#'                            maxit = iters)
#' # Try JerkianMLP
#' sens <- NeuralSens::JerkianMLP(nnetmod, trData = nntrData, plot = FALSE)
#' sens
#' @method print JerkMLP
#' @export
print.JerkMLP <- function(x, n = 5, round_digits = NULL, ...) {
  cat("Sensitivity analysis of ", paste(x$mlp_struct, collapse = "-"), " MLP network.\n", sep = "")
  # cat("Measures are calculated using the partial derivatives of ", x$layer_end, " layer's ",
  #     ifelse(x$layer_end_input,"input","output"), "\nwith respect to ", x$layer_origin, " layer's ",
  #     ifelse(x$layer_origin_input,"input","output"),".\n",
  #     sep = "")
  cat("\n  ",nrow(x$trData)," samples\n\n", sep = "")
  cat("Sensitivities of each output (only ",min(n,dim(x$raw_sens[[1]])[3])," first samples):\n", sep = "")
  if (!is.null(round_digits)) {
    if (round_digits >= 0) {
      x$raw_sens <- lapply(x$raw_sens,
                       function(y) {
                        round(y, round_digits)
                       })
    }
  }
  for (out in 1:length(x$raw_sens)) {
    cat("$",names(x$raw_sens)[out],"\n", sep = "")
    invisible(print(x$raw_sens[[out]][,,1:min(n,dim(x$raw_sens[[1]])[3])]))
  }
}
#' Plot method for the JerkMLP Class
#'
#' Plot the sensitivities and sensitivity metrics of a \code{JerkMLP} object.
#' @param x \code{JerkMLP} object created by \code{\link[NeuralSens]{JerkianMLP}}
#' @param plotType \code{character} specifying which type of plot should be created. It can be:
#' \itemize{
#'      \item "sensitivities" (default): use \code{\link[NeuralSens]{JerkianMLP}} function
#'      \item "time": use \code{\link[NeuralSens]{SensTimePlot}} function
#'      \item "features": use  \code{\link[NeuralSens]{HessFeaturePlot}} function
#'      \item "matrix": use \code{\link[NeuralSens]{SensMatPlot}} function to show the values
#'      of second partial derivatives
#'      \item "interactions": use \code{\link[NeuralSens]{SensMatPlot}} function to show the
#'      values of second partial derivatives and the first partial derivatives in the diagonal
#'      }
#' @param ... additional parameters passed to plot function of the \code{NeuralSens} package
#' @return list of graphic objects created by \code{\link[ggplot2]{ggplot}}
#' @examples
#' #' ## Load data -------------------------------------------------------------------
#' data("DAILY_DEMAND_TR")
#' fdata <- DAILY_DEMAND_TR
#'
#' ## Parameters of the NNET ------------------------------------------------------
#' hidden_neurons <- 5
#' iters <- 250
#' decay <- 0.1
#'
#' ################################################################################
#' #########################  REGRESSION NNET #####################################
#' ################################################################################
#' ## Regression dataframe --------------------------------------------------------
#' # Scale the data
#' fdata.Reg.tr <- fdata[,2:ncol(fdata)]
#' fdata.Reg.tr[,3] <- fdata.Reg.tr[,3]/10
#' fdata.Reg.tr[,1] <- fdata.Reg.tr[,1]/1000
#'
#' # Normalize the data for some models
#' preProc <- caret::preProcess(fdata.Reg.tr, method = c("center","scale"))
#' nntrData <- predict(preProc, fdata.Reg.tr)
#'
#' #' ## TRAIN nnet NNET --------------------------------------------------------
#' # Create a formula to train NNET
#' form <- paste(names(fdata.Reg.tr)[2:ncol(fdata.Reg.tr)], collapse = " + ")
#' form <- formula(paste(names(fdata.Reg.tr)[1], form, sep = " ~ "))
#'
#' set.seed(150)
#' nnetmod <- nnet::nnet(form,
#'                            data = nntrData,
#'                            linear.output = TRUE,
#'                            size = hidden_neurons,
#'                            decay = decay,
#'                            maxit = iters)
#' # Try JerkianMLP
#' sens <- NeuralSens::JerkianMLP(nnetmod, trData = nntrData, plot = FALSE)
#' \donttest{
#' plot(sens)
#' plot(sens,"time")
#' }
#' @method plot JerkMLP
#' @export
plot.JerkMLP <- function(x,
                         plotType = c("sensitivities","time", "features","matrix","interactions"),
                         ...) {
  plotType <- match.arg(plotType)

  switch(plotType,
         sensitivities = {
           NeuralSens::SensitivityPlots(x, ...)
         },
         time = {
           NeuralSens::SensTimePlot(x, ...)
         },
         features = {
           NeuralSens::HessFeaturePlot(x, ...)
         },
         matrix = {
           NeuralSens::SensMatPlot(x, senstype = "matrix", ...)
         },
         interactions = {
           NeuralSens::SensMatPlot(x, senstype = "interactions", ...)
         })
}
#' Convert a JerkMLP to a SensMLP object
#'
#' Auxiliary function to turn a JerkMLP object to a SensMLP object in order to use the plot-related functions
#' associated with SensMLP
#' @param x \code{JerkMLP} object
#' @return \code{SensMLP} object
#' @export
HessToSensMLP <- function(x) {
  y <- x
  for (out in 1:length(x$sens)) {
    y$sens[[out]]$mean <- as.vector(x$sens[[out]]$mean)
    y$sens[[out]]$std <- as.vector(x$sens[[out]]$std)
    y$sens[[out]]$meanSensSQ <- as.vector(x$sens[[out]]$meanSensSQ)
    y$sens[[out]] <- as.data.frame.list(y$sens[[out]])
    rownames(y$sens[[out]]) <- apply(expand.grid(dimnames(x$sens[[out]]$mean)[[1]],
                                                 dimnames(x$sens[[out]]$mean)[[2]],
                                                 stringsAsFactors = FALSE),
                                     1,paste,collapse = "_")
    indx <- matrix(TRUE, nrow(x$sens[[out]][[1]]), nrow(x$sens[[out]][[1]]))
    indx[upper.tri(indx)] <- FALSE
    y$raw_sens[[out]] <- data.matrix(data.frame(aperm(y$raw_sens[[out]],c(3,1,2))))
    colnames(y$raw_sens[[out]]) <- rownames(y$sens[[out]])
    y$sens[[out]] <- y$sens[[out]][as.vector(indx),]
    y$raw_sens[[out]] <- y$raw_sens[[out]][,as.vector(indx)]
  }
  class(y) <- "SensMLP"
  return(y)
}
