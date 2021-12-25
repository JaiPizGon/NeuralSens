#' Feature sensitivity plot
#'
#' @description Show the distribution of the sensitivities of the output
#' in \code{geom_sina()} plot which color depends on the input values
#' @param object fitted neural network model or \code{array} containing the raw
#' sensitivities from the function \code{\link[NeuralSens]{SensAnalysisMLP}}
#' @param fdata \code{data.frame} containing the data to evaluate the sensitivity of the model.
#' Not needed if the raw sensitivities has been passed as \code{object}
#' @param ... further arguments that should be passed to  \code{\link[NeuralSens]{SensAnalysisMLP}} function
#' @return list of Feature sensitivity plot as described in
#' \url{https://www.r-bloggers.com/2019/03/a-gentle-introduction-to-shap-values-in-r/}
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
#' hess <- NeuralSens::HessianMLP(nnetmod, trData = nntrData, plot = FALSE)
#' NeuralSens::HessFeaturePlot(hess)
#' @export HessFeaturePlot
HessFeaturePlot <- function(object, fdata = NULL, ...) {
  # Check if the object passed is a model or the sensitivities
  if(!is.HessMLP(object)) {
    # Check if fdata has been passed to the function to calculate sensitivities
    if (is.null(fdata)) {
      stop("Must be passed fdata to calculate sensitivities of the model")
    }
    # Obtain raw sensitivities
    HessMLP <- NeuralSens::HessianMLP(object,
                                     trData = fdata,
                                     plot = FALSE,
                                     ...)
    rawSens <- HessMLP$raw_sens

  } else if(is.HessMLP(object)){
    # The raw sensitivities has been passed instead of the model
    HessMLP <- object
    rawSens <- HessMLP$raw_sens
    fdata <- HessMLP$trData
  } else {
    stop(paste0("Class ", class(object)," is not accepted as object"))
  }
  # Confirm that fdata is a data.frame due to incompatibilities issues later
  trData <- as.data.frame(fdata)
  if (any(sapply(trData, function(x){is.factor(x)||is.character(x)}))) {
    # Create dummies if there are factors
    dumData <- fastDummies::dummy_columns(trData[,sapply(trData, function(x){is.factor(x)||is.character(x)})])
    # If there is only one factor, substitute ".data" by name of variable
    names(dumData) <- stringr::str_replace(names(dumData), ".data",
                                           names(trData)[sapply(trData,function(x){is.factor(x)||is.character(x)})])
    for (i in 1:sum(sapply(trData,function(x){is.factor(x)||is.character(x)}))) {
      trData[,names(trData) == names(dumData)[i]] <- NULL
      trData <- cbind(trData, dumData[,grepl(paste0(names(dumData)[i],"_"),names(dumData))])
      names(trData)[grepl(paste0(names(dumData)[i],"_"),names(trData))] <- paste0(names(dumData)[i],unique(dumData[,i]))
    }
  }

  # if (!((object$layer_origin == 1) && object$layer_origin_input)) {
  #   stop("Feature plots only available for SensAnalysisMLP objects with origin layer the input layer")
  # }

  trData <- trData[,colnames(rawSens[[1]])]
  # Normalize data between 0 and 1
  # trData <- as.data.frame(lapply(trData[,colnames(rawSens[[1]])],
  #                                function(x){
  #                                  (x - min(x, na.rm = T)) /
  #                                    (max(x, na.rm = T) - min(x, na.rm = T))
  #                                  }))
  # Create plot layer by layer
  plotlist <- list()
  plot_for_input2 <- function(trData, rawSens, out, i, j) {
    p3 <- ggplot2::ggplot() + ggplot2::geom_vline(xintercept = 0) +
      ggplot2::geom_point(ggplot2::aes(x = rawSens[[out]][i,j,], y = trData[,i],
                                       color = trData[,j])) +
      ggplot2::scale_color_gradient(low="#FFCC33", high="#6600CC") +
      ggplot2::labs(x = "Second derivative", y = names(trData)[i], color = names(trData)[j]) +
      ggplot2::theme(legend.position = "bottom")
    return(p3)
  }
  plot_for_input <- function(trData, rawSens, out, i) {
    xlimits <- c(0,0)
    p2 <- list()
    for (j in 1:ncol(trData)) {
      p2[[j]] <- plot_for_input2(trData, rawSens, out, i, j)
      xlimits[1] <- min(c(xlimits[1],min(rawSens[[out]][i,j,])))
      xlimits[2] <- max(c(xlimits[2],max(rawSens[[out]][i,j,])))
    }
    p2 <- lapply(p2, function(x){x + ggplot2::xlim(xlimits)})
    return(p2)
  }
  plot_for_output <- function(trData, rawSens, out) {
    p <- list()
    for (i in 1:ncol(trData)) {
      p <- c(p,plot_for_input(trData, rawSens, out, i))
    }
    return(invisible(gridExtra::grid.arrange(grobs = p, ncol = ncol(trData))))
  }

  for (out in 1:length(rawSens)) {
      plotlist[[out]] <- plot_for_output(trData, rawSens, out)
  }
  return(invisible(plotlist))
}
