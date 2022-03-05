#' Plot sensitivities of a neural network model
#'
#' @description Function to plot the sensitivities created by \code{\link[NeuralSens]{SensAnalysisMLP}}.
#' @param sens \code{SensAnalysisMLP} object created by \code{\link[NeuralSens]{SensAnalysisMLP}}.
#' @return \code{SensAnalysisMLP} object with the sensitivities calculated
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
ComputeHessMeasures <- function(sens) {
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

  # Compute the cumulative second derivatives
  X <- list()
  Q <- list()
  D_ <- list()

  # Initialize the cross-derivatives
  D <- sens$layer_derivatives
  D2 <- sens$layer_second_derivatives
  W <- sens$mlp_wts
  D_[[1]] <- sens$layer_derivatives[[sens_origin_layer]]
  Q[[1]] <- diag3Darray(dim = mlpstr[sens_origin_layer])
  X[[1]] <- D2[[sens_origin_layer]]

  l <- 1
  if (sens_origin_layer != sens_end_layer) {
    # Damn, there are no array multiplications, we need to use sapplys
    counter <- 1
    for (l in (sens_origin_layer+1):sens_end_layer) {
      counter <- counter + 1
      # Now we add a third dimension for the second input
      D_[[counter]] <- array(NA, dim=c(mlpstr[sens_origin_layer], mlpstr[l], nrow(TestData)))
      Q[[counter]] <- array(NA, dim=c(mlpstr[sens_origin_layer], mlpstr[l], mlpstr[sens_origin_layer], nrow(TestData)))
      X[[counter]] <- array(NA, dim=c(mlpstr[sens_origin_layer], mlpstr[l], mlpstr[sens_origin_layer], nrow(TestData)))
      for (irow in 1:nrow(TestData)) {
        D_[[counter]][,,irow] <- matrix(D_[[counter - 1]][,,irow,drop=FALSE],
                                        nrow=dim(D_[[counter - 1]][,,irow,drop=FALSE])[1],
                                        ncol=dim(D_[[counter - 1]][,,irow,drop=FALSE])[2]) %*%
          matrix(D[[l - 1]][,,irow,drop=FALSE],
                 nrow=dim(D[[l - 1]][,,irow,drop=FALSE])[1],
                 ncol=dim(D[[l - 1]][,,irow,drop=FALSE])[2]) %*%
          matrix(W[[l]][2:nrow(W[[l]]),,drop=FALSE],
                 nrow = dim(W[[l]])[1] - 1,
                 ncol = dim(W[[l]])[2])

        Q[[counter]][,,,irow] <- array(apply(array(X[[counter - 1]][,,,irow,drop=FALSE], dim = dim(X[[counter - 1]])[1:3]), 3,
                                             function(x) x %*% matrix(W[[l]][2:nrow(W[[l]]),,drop=FALSE],
                                                                      nrow = dim(W[[l]])[1] - 1,
                                                                      ncol = dim(W[[l]])[2])),
                                       dim = c(mlpstr[sens_origin_layer], dim(W[[l]])[2], mlpstr[sens_origin_layer]))

        X[[counter]][,,,irow] <- array(apply(array(apply(array(D2[[l]][,,,irow], dim = dim(D2[[l]])[1:3]), 3,
                                                         function(x) matrix(D_[[counter]][,,irow], nrow = dim(D_[[counter]])[1]) %*% x),
                                                   dim = c(mlpstr[sens_origin_layer], dim(D2[[l]])[2], dim(D2[[l]])[3])),
                                             1, function(x) matrix(D_[[counter]][,,irow], nrow = dim(D_[[counter]])[1]) %*% x),
                                       dim = c(mlpstr[sens_origin_layer], dim(D2[[l]])[2], mlpstr[sens_origin_layer])) + # Here ends y^2/z^2 * z/x1 * z/x2
          array(apply(array(Q[[counter]][,,,irow],dim = dim(Q[[counter]])[1:3]),3,
                      function(x){x %*% D[[l]][,,irow]}),
                dim = c(mlpstr[sens_origin_layer], dim(D2[[l]])[2], mlpstr[sens_origin_layer]))
      }
    }
    l <- counter
  }

  if (sens_end_input) {
    der <- Q[[l]]
  } else {
    der <- X[[l]]
  }
  # Prepare the derivatives for the following calculations
  varnames <- sens$coefnames
  if (sens_origin_layer != 1) {
    varnames <- paste0("Neuron ",sens_origin_layer,".",1:mlpstr[sens_origin_layer])
  }
  dimnames(der)[[1]] <- varnames
  dimnames(der)[[3]] <- varnames
  der <- aperm(der, c(1,3,2,4))

  # Add rawSens to the structure
  rs <- list()
  out <- list()
  for (i in 1:dim(der)[3]) {
    out[[i]] <- list(
      mean = apply(der[,,i,], c(1,2), mean, na.rm = TRUE),
      std = apply(der[,,i,], c(1,2), stats::sd, na.rm = TRUE),
      meanSensSQ = sqrt(apply(der[,,i,]^2, c(1,2), mean, na.rm = TRUE))
    )
    rs[[i]] <- der[,,i,]
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
