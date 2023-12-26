#' Plot sensitivities of a neural network model
#'
#' @description Function to plot the sensitivities created by \code{\link[NeuralSens]{SensAnalysisMLP}}.
#' @param sens \code{SensAnalysisMLP} object created by \code{\link[NeuralSens]{SensAnalysisMLP}}.
#' @return \code{SensAnalysisMLP} object with the sensitivities calculated
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
ComputeSensMeasures <- function(sens) {
  mlpstr <- sens$mlp_struct
  TestData <- sens$trData
  sens_origin_layer<- sens$layer_origin
  sens_end_layer <- sens$layer_end
  sens_origin_input <- sens$layer_origin_input
  sens_end_input <- sens$layer_end_input

  # Check all sensitivity arguments makes sense
  # Check which is the output we want the derivWative of
  if (sens_end_layer == "last") {
    # User wants the derivative of the last layer of the neural network
    sens_end_layer = length(mlpstr)
  }

  # Detect that origin and end layer are defined by a number
  if (!is.numeric(c(sens_end_layer,sens_origin_layer))) {
    stop("End layer and origin layer must be specified by a strictly positive number")
  }
  # Detect that layers are specified by strictly positive numbers
  if (any(c(sens_end_layer,sens_origin_layer) <= 0)) {
    stop("End layer and origin layer must be specified by a strictly positive number")
  }
  # Detect that at least there is one layer between origin and end of derivatives
  if ((sens_end_layer < sens_origin_layer) ||
      ((sens_end_layer == sens_origin_layer) &&
       !(sens_origin_input && !sens_end_input))) {
    stop("There must be at least one layer between end and origin")
  }

  # Detect that exists the layers specified
  if (sens_end_layer > length(mlpstr)) {
    stop("The layers specified could not be found in the neural network model")
  }

  # Compute the cummulative derivatives
  D <- list()
  D[[1]] <- array(diag(mlpstr[sens_origin_layer]),
                  dim=c(mlpstr[sens_origin_layer],
                        mlpstr[sens_origin_layer],
                        nrow(TestData)))
  if (sens_origin_input) {
    D[[1]] <- sens$layer_derivatives[[sens_origin_layer]]
  }
  l <- 1
  # Only perform further operations if origin is not equal to end layer
  if (sens_origin_layer != sens_end_layer) {
    counter <- 1
    for (l in (sens_origin_layer+1):sens_end_layer) {
      counter <- counter + 1
      D[[counter]] <- array(NA, dim=c(mlpstr[sens_origin_layer], mlpstr[l], nrow(TestData)))
      # Check if it must be multiplied by the jacobian of the nest layer
      if ((l == sens_end_layer) && sens_end_input) {
        for (irow in 1:nrow(TestData)){
          D[[counter]][,,irow] <- matrix(D[[counter - 1]][,,irow,drop=FALSE],
                                         nrow = dim(D[[counter - 1]])[1],
                                         ncol = dim(D[[counter - 1]])[2]) %*%
            sens$mlp_wts[[l]][2:nrow(sens$mlp_wts[[l]]),,drop=FALSE]
        }
      } else {
        for (irow in 1:nrow(TestData)){
          D[[counter]][,,irow] <- matrix(D[[counter - 1]][,,irow,drop=FALSE],
                                         nrow = dim(D[[counter - 1]])[1],
                                         ncol = dim(D[[counter - 1]])[2])  %*%
            sens$mlp_wts[[l]][2:nrow(sens$mlp_wts[[l]]),,drop=FALSE] %*%
            matrix(sens$layer_derivatives[[l]][,,irow,drop=FALSE],
                   nrow = dim(sens$layer_derivatives[[l]])[1],
                   ncol = dim(sens$layer_derivatives[[l]])[2])
        }
      }
    }
    l <- counter
  }
  der <- aperm(D[[l]],c(3,1,2))

  # Prepare the derivatives for the following calculations
  varnames <- sens$coefnames
  if (sens_origin_layer != 1) {
    varnames <- paste0("Neuron ",sens_origin_layer,".",1:mlpstr[sens_origin_layer])
  }
  colnames(der) <- varnames

  # Add rawSens to the structure
  rs <- list()
  out <- list()
  for (i in 1:dim(der)[3]) {
    out[[i]] <- data.frame(
      mean = colMeans(der[, , i, drop=FALSE], na.rm = TRUE),
      std = apply(der[, , i, drop=FALSE], 2, stats::sd, na.rm = TRUE),
      meanSensSQ = sqrt(colMeans(der[, , i, drop=FALSE] ^ 2, na.rm = TRUE)),
      row.names = varnames
    )
    rs[[i]] <- der[,,i]
  }
  if (is.factor(sens$trData$.outcome)) {
    names(out) <- make.names(unique(sens$trData$.outcome), unique = TRUE)[1:length(out)]
  } else if (!is.null(sens$output_name)) {
    names(out) <- sens$output_name
  }
  names(rs) <- names(out)
  sens$sens <- out
  sens$raw_sens <- rs
  return(sens)
}
