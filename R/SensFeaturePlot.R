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
#' \url{https://www.r-bloggers.com/a-gentle-introduction-to-shap-values-in-r/}
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
#' NeuralSens::SensFeaturePlot(sens)
#' @export SensFeaturePlot
SensFeaturePlot <- function(object, fdata = NULL, ...) {
  # Check if the object passed is a model or the sensitivities
  if (!is.SensMLP(object)) {
    # Check if fdata has been passed to the function to calculate sensitivities
    if (is.null(fdata)) {
      stop("Must be passed fdata to calculate sensitivities of the model")
    }
    # Obtain raw sensitivities
    SensMLP <- NeuralSens::SensAnalysisMLP(object,
                                           trData = fdata,
                                           plot = FALSE,
                                           ...)
    rawSens <- SensMLP$raw_sens

  } else if(is.SensMLP(object)){
    # The raw sensitivities has been passed instead of the model
    SensMLP <- object
    rawSens <- SensMLP$raw_sens
    fdata <- SensMLP$trData
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

  # Normalize data between 0 and 1
  trData <- as.data.frame(lapply(trData[,colnames(rawSens[[1]])],
                                 function(x){
                                   (x - min(x, na.rm = T)) /
                                     (max(x, na.rm = T) - min(x, na.rm = T))
                                   }))
  # Create plot layer by layer
  plotlist <- list()
  for (out in 1:length(rawSens)) {
    local({
      out <- out
      p <- ggplot2::ggplot()
      for (i in 1:ncol(trData)) {
        local({
          i <- i
          p <<- p +
            ggplot2::geom_vline(xintercept = i, linetype = "dotdash") +
            ggplot2::geom_violin(alpha = 0.3,
                                 ggplot2::aes(x = i, y = rawSens[[out]][,i]),
                                 color = "darkgray", fill = "gray") +
            ggplot2::geom_hline(yintercept = 0) +
            ggforce::geom_sina(ggplot2::aes(x = i, y = rawSens[[out]][,i],
                                            color = trData[,i]),
                               scale = FALSE)
          # ggplot2::geom_jitter(ggplot2::aes(x = i, y = rawSens[,i,1], color = trData[,i]),
          #                    shape = 16, position = position_jitterdodge(),
          #                    size = 1.5, alpha = 0.7) +
        })
      }
      p <- p +
        ggplot2::labs(x = NULL, y = "sens") +
        ggplot2::coord_flip() +
        ggplot2::scale_x_continuous(breaks = seq_len(dim(rawSens[[out]])[2]),
                                    labels = colnames(rawSens[[out]])) +
        ggplot2::scale_color_gradient(low="#FFCC33", high="#6600CC",
                                      breaks=c(0,1), labels=c("Low","High")) +
        ggplot2::labs(color = "Feature value") +
        ggplot2::theme(legend.position = "bottom")
      print(p)
      plotlist[[out]] <<- p
    })
  }
  return(invisible(plotlist))
}
