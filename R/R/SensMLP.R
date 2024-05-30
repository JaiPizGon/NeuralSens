#' Constructor of the SensMLP Class
#'
#' Create an object of SensMLP class
#' @param sens \code{list} of sensitivity measures, one \code{data.frame} per output neuron
#' @param raw_sens \code{list} of sensitivities, one \code{matrix} per output neuron
#' @param mlp_struct \code{numeric} vector describing the structur of the MLP model
#' @param trData \code{data.frame} with the data used to calculate the sensitivities
#' @param coefnames \code{character} vector with the name of the predictor(s)
#' @param output_name \code{character} vector with the name of the output(s)
#' @param cv \code{list} list with critical values of significance for std and mean square.
#' @param boot \code{array} bootstrapped sensitivity measures.
#' @param boot.alpha \code{array} significance level.
#' Defaults to \code{NULL}. Only available for analyzed \code{caret::train} models.
#' @return \code{SensMLP} object
#' @references
#' Pizarroso J, Portela J, Muñoz A (2022). NeuralSens: Sensitivity Analysis of
#' Neural Networks. Journal of Statistical Software, 102(7), 1-36.
#' @export SensMLP
SensMLP <- function(sens = list(),
                    raw_sens = list(),
                    mlp_struct = numeric(),
                    trData = data.frame(),
                    coefnames = character(),
                    output_name = character(),
                    cv = NULL,
                    boot = NULL,
                    boot.alpha = NULL
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
      output_name = output_name,
      cv = cv,
      boot = boot,
      boot.alpha = boot.alpha
    ),
    class = "SensMLP"
  )
}
#' Check if object is of class \code{SensMLP}
#'
#' Check if object is of class \code{SensMLP}
#' @param object \code{SensMLP} object
#' @return \code{TRUE} if \code{object} is a \code{SensMLP} object
#' @references
#' Pizarroso J, Portela J, Muñoz A (2022). NeuralSens: Sensitivity Analysis of
#' Neural Networks. Journal of Statistical Software, 102(7), 1-36.
#' @export is.SensMLP
is.SensMLP <- function(object) {
  any(class(object) == "SensMLP")
}
#' Summary Method for the SensMLP Class
#'
#' Print the sensitivity metrics of a \code{SensMLP} object.
#' This metrics are the mean sensitivity, the standard deviation
#' of sensitivities and the mean of sensitivities square
#' @param object \code{SensMLP} object created by \code{\link[NeuralSens]{SensAnalysisMLP}}
#' @param ... additional parameters
#' @return summary object of the \code{SensMLP} object passed
#' @references
#' Pizarroso J, Portela J, Muñoz A (2022). NeuralSens: Sensitivity Analysis of
#' Neural Networks. Journal of Statistical Software, 102(7), 1-36.
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
  print(object, ...)
}
#' Print method of the summary SensMLP Class
#'
#' Print the sensitivity metrics of a \code{SensMLP} object.
#' This metrics are the mean sensitivity, the standard deviation
#' of sensitivities and the mean of sensitivities square
#' @param x \code{summary.SensMLP} object created by summary method of \code{SensMLP} object
#' @param round_digits \code{integer} number of decimal places, default \code{NULL}
#' @param boot.alpha \code{float} significance level to show statistical metrics. If \code{NULL},
#' boot.alpha inherits from \code{x} is used. Defaults to \code{NULL}.
#' @param ... additional parameters
#' @references
#' Pizarroso J, Portela J, Muñoz A (2022). NeuralSens: Sensitivity Analysis of
#' Neural Networks. Journal of Statistical Software, 102(7), 1-36.
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
print.summary.SensMLP <- function(x, round_digits = NULL, boot.alpha = NULL, ...) {
  cat("Sensitivity analysis of ", paste(x$mlp_struct, collapse = "-"), " MLP network.\n\n", sep = "")
  if (!is.null(x$cv)) {
    if (!is.null(boot.alpha)) {
      x <- NeuralSens::ChangeBootAlpha(x, boot.alpha)
    }
    cat(paste0("Bootstrapped sensitivity measures with significance level \u03B1=", as.character(x$boot.alpha), ". \n"))
    cat(paste0("Bootstrapped metrics with ", as.character(dim(x$boot)[3]), " repetitions. \n\n"))
    x$sens[[1]][,"sd.mean"] <- apply(x$boot[,1,], 1, stats::sd)
  }
  # cat("Measures are calculated using the partial derivatives of ", x$layer_end, " layer's ",
  #     ifelse(x$layer_end_input,"input","output"), "\nwith respect to ", x$layer_origin, " layer's ",
  #     ifelse(x$layer_origin_input,"input","output"),".\n\n",
  #     sep = "")
  if (!is.null(round_digits)) {
    if (round_digits >= 0) {
      x$sens <- lapply(x$sens,
                       function(y) {
                         as.data.frame(lapply(y, function(z){
                           round(z, round_digits)
                         }), row.names = rownames(y), col.names=colnames(y))
                       })
    }
  }

  if (!is.null(x$cv)) {
    x$sens[[1]][,"sd.mean"] <- paste0("\u00B1", as.character(x$sens[[1]][,"sd.mean"]))
    x$sens[[1]][,"linearity"] <- ifelse(x$cv[[1]]$signif, "non-linear", "linear")
    x$sens[[1]][,"signif."] <- ifelse(x$cv[[2]]$signif, 1, 0)
    x$sens[[1]] <- x$sens[[1]][,c("mean", "sd.mean", "std", "linearity", "meanSensSQ", "signif.")]
    colnames(x$sens[[1]]) <- c("mean", "\u00B1mean", "std", "linearity", "meanSensSQ", "signif.")
  }

  cat("Sensitivity measures of each output:\n")
  invisible(print(x$sens))
}

#' Change significance of boot SensMLP Class
#'
#' For a SensMLP Class object, change the significance level of the statistical tests
#' @param x \code{SensMLP} object created by \code{\link[NeuralSens]{SensAnalysisMLP}}
#' @param boot.alpha \code{float} significance level
#' @return \code{SensMLP} object with changed significance level. All boot related
#' metrics are changed
#' @examples
#' \donttest{
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
#'
#' ## TRAIN nnet NNET --------------------------------------------------------
#'
#' set.seed(150)
#' nnetmod <- caret::train(DEM ~ .,
#'                  data = fdata.Reg.tr,
#'                  method = "nnet",
#'                  tuneGrid = expand.grid(size = c(1), decay = c(0.01)),
#'                  trControl = caret::trainControl(method="none"),
#'                  preProcess = c('center', 'scale'),
#'                  linout = FALSE,
#'                  trace = FALSE,
#'                  maxit = 300)
#' # Try SensAnalysisMLP
#' sens <- NeuralSens::SensAnalysisMLP(nnetmod, trData = fdata.Reg.tr,
#'                                     plot = FALSE, boot.R=2, output_name='DEM')
#' NeuralSens::ChangeBootAlpha(sens, boot.alpha=0.1)
#' }
#' @export ChangeBootAlpha
ChangeBootAlpha <- function(x, boot.alpha) {
  # Configuration of significance test
  num_hypotheses <- nrow(x$sens[[1]])

  # Obtain Tnj
  Tnj <- data.matrix(x$sens[[1]])

  # Obtain Tnj_b
  Tnj_b <- x$boot

  # Statistical test of std and mean square
  cv <- list()
  for (i in 1:2) {
    cv[[i]] <- NeuralSens::kStepMAlgorithm(
      bootstrap_stats = data.matrix(aperm(Tnj_b, c(3,2,1))[,i+1,]),
      original_stats = Tnj[,i+1],
      num_hypotheses = num_hypotheses,
      alpha = boot.alpha,
      k = 1)
  }
  x$cv <- cv
  x$boot.alpha <- boot.alpha
  return(x)
}
#' Print method for the SensMLP Class
#'
#' Print the sensitivities of a \code{SensMLP} object.
#' @param x \code{SensMLP} object created by \code{\link[NeuralSens]{SensAnalysisMLP}}
#' @param n \code{integer} specifying number of sensitivities to print per each output
#' @param round_digits \code{integer} number of decimal places, default \code{NULL}
#' @param ... additional parameters
#' @references
#' Pizarroso J, Portela J, Muñoz A (2022). NeuralSens: Sensitivity Analysis of
#' Neural Networks. Journal of Statistical Software, 102(7), 1-36.
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
print.SensMLP <- function(x, n = 5, round_digits = NULL, ...) {
  cat("Sensitivity analysis of ", paste(x$mlp_struct, collapse = "-"), " MLP network.\n", sep = "")
  # cat("Measures are calculated using the partial derivatives of ", x$layer_end, " layer's ",
  #     ifelse(x$layer_end_input,"input","output"), "\nwith respect to ", x$layer_origin, " layer's ",
  #     ifelse(x$layer_origin_input,"input","output"),".\n",
  #     sep = "")
  cat("\n  ",nrow(x$trData)," samples\n\n", sep = "")
  cat("Sensitivities of each output (only ",min(n,nrow(x$raw_sens[[1]]))," first samples):\n", sep = "")
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
    invisible(print(x$raw_sens[[out]][1:min(n,nrow(x$raw_sens[[1]])),]))
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
#' @references
#' Pizarroso J, Portela J, Muñoz A (2022). NeuralSens: Sensitivity Analysis of
#' Neural Networks. Journal of Statistical Software, 102(7), 1-36.
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
#' \donttest{
#' plot(sens)
#' plot(sens,"time")
#' plot(sens,"features")
#' }
#' @method plot SensMLP
#' @export
plot.SensMLP <- function(x,
                         plotType = c("sensitivities","time","features"),
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
           NeuralSens::SensFeaturePlot(x, ...)
         })
}
