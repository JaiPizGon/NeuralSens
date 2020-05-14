#' Constructor of the HessMLP Class
#'
#' Create an object of HessMLP class
#' @param sens \code{list} of sensitivity measures, one \code{list} per output neuron
#' @param raw_sens \code{list} of sensitivities, one \code{array} per output neuron
#' @param mlp_struct \code{numeric} vector describing the structur of the MLP model
#' @param trData \code{data.frame} with the data used to calculate the sensitivities
#' @param coefnames \code{character} vector with the name of the predictor(s)
#' @param output_name \code{character} vector with the name of the output(s)
#' @return \code{HessMLP} object
#' @export HessMLP
HessMLP <- function(sens = list(),
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
    class = "HessMLP"
  )
}
#' Check if object is of class \code{HessMLP}
#'
#' Check if object is of class \code{HessMLP}
#' @param object \code{HessMLP} object
#' @return \code{TRUE} if \code{object} is a \code{HessMLP} object
#' @export is.HessMLP
is.HessMLP <- function(object) {
  any(class(object) == "HessMLP")
}
#' Summary Method for the HessMLP Class
#'
#' Print the sensitivity metrics of a \code{HessMLP} object.
#' This metrics are the mean sensitivity, the standard deviation
#' of sensitivities and the mean of sensitivities square
#' @param object \code{HessMLP} object created by \code{\link[NeuralSens]{HessianMLP}}
#' @param ... additional parameters
#' @return summary object of the \code{HessMLP} object passed
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
#' # Try HessianMLP
#' sens <- NeuralSens::HessianMLP(nnetmod, trData = nntrData, plot = FALSE)
#' summary(sens)
#' @method summary HessMLP
#' @export
summary.HessMLP <- function(object, ...) {
  class(object) <- c("summary.HessMLP", class(object))
  object
}
#' Print method of the summary HessMLP Class
#'
#' Print the sensitivity metrics of a \code{HessMLP} object.
#' This metrics are the mean sensitivity, the standard deviation
#' of sensitivities and the mean of sensitivities square
#' @param x \code{summary.HessMLP} object created by summary method of \code{HessMLP} object
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
#' # Try HessianMLP
#' sens <- NeuralSens::HessianMLP(nnetmod, trData = nntrData, plot = FALSE)
#' print(summary(sens))
#' @method print summary.HessMLP
#' @export
print.summary.HessMLP <- function(x, ...) {
  cat("Hessian matrix of ", paste(x$mlp_struct, collapse = "-"), " MLP network.\n\n", sep = "")
  # cat("Measures are calculated using the partial derivatives of ", x$layer_end, " layer's ",
  #     ifelse(x$layer_end_input,"input","output"), "\nwith respect to ", x$layer_origin, " layer's ",
  #     ifelse(x$layer_origin_input,"input","output"),".\n\n",
  #     sep = "")
  cat("Hessian measures of each output:\n")
  invisible(print(x$sens))
}

#' Print method for the HessMLP Class
#'
#' Print the sensitivities of a \code{HessMLP} object.
#' @param x \code{HessMLP} object created by \code{\link[NeuralSens]{HessianMLP}}
#' @param n \code{integer} specifying number of sensitivities to print per each output
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
#' # Try HessianMLP
#' sens <- NeuralSens::HessianMLP(nnetmod, trData = nntrData, plot = FALSE)
#' sens
#' @method print HessMLP
#' @export
print.HessMLP <- function(x, n = 5, ...) {
  cat("Sensitivity analysis of ", paste(x$mlp_struct, collapse = "-"), " MLP network.\n", sep = "")
  # cat("Measures are calculated using the partial derivatives of ", x$layer_end, " layer's ",
  #     ifelse(x$layer_end_input,"input","output"), "\nwith respect to ", x$layer_origin, " layer's ",
  #     ifelse(x$layer_origin_input,"input","output"),".\n",
  #     sep = "")
  cat("\n  ",nrow(x$trData)," samples\n\n", sep = "")
  cat("Sensitivities of each output (only ",min(n,dim(x$raw_sens[[1]])[3])," first samples):\n", sep = "")
  for (out in 1:length(x$raw_sens)) {
    cat("$",names(x$raw_sens)[out],"\n", sep = "")
    invisible(print(x$raw_sens[[out]][,,1:min(n,dim(x$raw_sens[[1]])[3])]))
  }
}
#' Plot method for the HessMLP Class
#'
#' Plot the sensitivities and sensitivity metrics of a \code{HessMLP} object.
#' @param x \code{HessMLP} object created by \code{\link[NeuralSens]{HessianMLP}}
#' @param plotType \code{character} specifying which type of plot should be created. It can be:
#' \itemize{
#'      \item "sensitivities" (default): use \code{\link[NeuralSens]{HessianMLP}} function
#'      \item "time": use \code{\link[NeuralSens]{SensTimePlot}} function
#'      \item "features": use  \code{\link[NeuralSens]{SensFeaturePlot}} function
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
#' # Try HessianMLP
#' sens <- NeuralSens::HessianMLP(nnetmod, trData = nntrData, plot = FALSE)
#' \dontrun{
#' plot(sens)
#' plot(sens,"time")
#' }
#' @method plot HessMLP
#' @export
plot.HessMLP <- function(x,
                         plotType = c("sensitivities","time","matrix","interactions"),
                         ...) {
  plotType <- match.arg(plotType)

  switch(plotType,
         sensitivities = {
           NeuralSens::SensitivityPlots(x, ...)
         },
         time = {
           NeuralSens::SensTimePlot(x, ...)
         },
         matrix = {
           NeuralSens::SensMatPlot(x, senstype = "matrix", ...)
         },
         interactions = {
           NeuralSens::SensMatPlot(x, senstype = "interactions", ...)
         })
}
#' Convert a HessMLP to a SensMLP object
#'
#' Auxiliary function to turn a HessMLP object to a SensMLP object in order to use the plot-related functions
#' associated with SensMLP
#' @param x \code{HessMLP} object
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
    y$raw_sens[[out]] <- data.matrix(data.frame(aperm(y$raw_sens[[out]],c(3,1,2))))
    colnames(y$raw_sens[[out]]) <- rownames(y$sens[[out]])
  }
  class(y) <- "SensMLP"
  return(y)
}
