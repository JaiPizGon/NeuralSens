#' Summary Method for the SensMLP Class
#'
#' Print the sensitivity metrics of a \code{SensMLP} object.
#' This metrics are the mean sensitivity, the standard deviation
#' of sensitivities and the mean of sensitivities square
#' @param object \code{SensMLP} object created by \code{\link[NeuralSens]{SensAnalysisMLP}}
#' @param ... additional parameters
#' @return summary object of the \code{SensMLP} object passed
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
#' # Try SensAnalysisMLP
#' sens <- NeuralSens::SensAnalysisMLP(nnetmod, trData = nntrData, plot = FALSE)
#' summary(sens)
#' @method summary SensMLP
#' @export
summary.SensMLP <- function(object, ...) {
  class(object) <- c("summary.SensMLP", class(object))
  object
}
#' Print method of the summary SensMLP Class
#'
#' Print the sensitivity metrics of a \code{SensMLP} object.
#' This metrics are the mean sensitivity, the standard deviation
#' of sensitivities and the mean of sensitivities square
#' @param x \code{summary.SensMLP} object created by summary method of \code{SensMLP} object
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
#' # Try SensAnalysisMLP
#' sens <- NeuralSens::SensAnalysisMLP(nnetmod, trData = nntrData, plot = FALSE)
#' print(summary(sens))
#' @method print summary.SensMLP
#' @export
print.summary.SensMLP <- function(x, ...) {
  cat("Sensitivity analysis of ", paste(x$mlp_struct, collapse = "-"), " MLP network.\n\n", sep = "")
  cat("Measures are calculated using the partial derivatives of ", x$layer_end, " layer's ",
      ifelse(x$layer_end_input,"input","output"), "\nwith respect to ", x$layer_origin, " layer's ",
      ifelse(x$layer_origin_input,"input","output"),".\n\n",
      sep = "")
  cat("Sensitivity measures of each output:\n")
  invisible(print(x$sens))
}

#' Print method for the SensMLP Class
#'
#' Print the sensitivities of a \code{SensMLP} object.
#' @param x \code{SensMLP} object created by \code{\link[NeuralSens]{SensAnalysisMLP}}
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
#' # Try SensAnalysisMLP
#' sens <- NeuralSens::SensAnalysisMLP(nnetmod, trData = nntrData, plot = FALSE)
#' sens
#' @method print SensMLP
#' @export
print.SensMLP <- function(x, ...) {
  cat("Sensitivity analysis of ", paste(x$mlp_struct, collapse = "-"), " MLP network.\n\n", sep = "")
  cat("Measures are calculated using the partial derivatives of ", x$layer_end, " layer's ",
      ifelse(x$layer_end_input,"input","output"), "\nwith respect to ", x$layer_origin, " layer's ",
      ifelse(x$layer_origin_input,"input","output"),".\n",
      sep = "")
  cat("\n  ",nrow(x$trData)," samples\n\n", sep = "")
  cat("Sensitivities of each output (only ",min(5,nrow(x$raw_sens[[1]]))," first samples):\n", sep = "")
  for (out in 1:length(x$raw_sens)) {
    cat("$",names(x$raw_sens)[out],"\n", sep = "")
    invisible(print(x$raw_sens[[out]][1:min(5,nrow(x$raw_sens[[1]])),]))
  }
}
#' Plot method for the SensMLP Class
#'
#' Plot the sensitivities and sensitivity metrics of a \code{SensMLP} object.
#' @param x \code{SensMLP} object created by \code{\link[NeuralSens]{SensAnalysisMLP}}
#' @param plotType \code{character} specifying which type of plot should be created. It can be:
#' \itemize{
#'      \item "sensitivities" (default): use \code{\link[NeuralSens]{SensAnalysisMLP}} function
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
#' # Try SensAnalysisMLP
#' sens <- NeuralSens::SensAnalysisMLP(nnetmod, trData = nntrData, plot = FALSE)
#' \dontrun{
#' plot(sens)
#' plot(sens,"time")
#' plot(sens,"features")
#' }
#' @method plot SensMLP
#' @export
plot.SensMLP <- function(x,
                         plotType = "sensitivities",
                         ...) {
  if (!(plotType %in% c("sensitivities", "time", "features"))) stop("plotType must be either sensitivities, time or features")

  switch(plotType,
         sensitivities = {
           NeuralSens::SensitivityPlots(x, ...)
         },
         time = {
           NeuralSens::SensTimePlot(x, ...)
         },
         features = {
           NeuralSens::SensFeaturePlot(x, ...)
         })
}
