#' Sensitivity analysis plot among time of the data
#'
#' @description Plot of sensitivity of the neural network output respect
#' to the inputs over the time variable from the data provided
#' @param object fitted neural network model or \code{array} containing the raw
#' sensitivities from the function \code{SensAnalysisMLP}
#' @param fdata \code{data.frame} containing the data to evaluate the sensitivity of the model.
#' Not needed if the raw sensitivities has been passed as \code{object}
#' @param date.var \code{Posixct vector} with the date of each sample of \code{fdata}
#' If \code{NULL}, the first variable with Posixct format of \code{fdata} is used as dates
#' @param facet \code{logical} if \code{TRUE}, function \code{facet_grid} from
#' @param ... arguments passed to the function to use S3 method
#' package \code{\link[ggplot2]{ggplot2-package}} to divide the plot
#' for each input variable.
#' @return \code{geom_line} plots for the inputs variables representing the
#' sensibility of each output respect to the inputs over time
#' @examples
#' ## Load data -------------------------------------------------------------------
#' data("DAILY_DEMAND_TR")
#' fdata <- as.data.frame(DAILY_DEMAND_TR)
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
#' fdata.Reg.tr[,2:3] <- fdata.Reg.tr[,2:3]/10
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
#' NeuralSens::SensTimePlot(nnetmod, fdata = nntrData, date.var = fdata[,1])
#' @export
#' @rdname SensTimePlot
SensTimePlot <- function(object, fdata = NULL, date.var = NULL, facet = FALSE) {
  # Check if the variable name of the date has been specified
  if (is.null(date.var)) {
    date.var <- fdata[,sapply(fdata, function(x){
      inherits(x,"POSIXct") || inherits(x,"POSIXlt")})]
  }
  # Check if the object passed is a model (list) or the raw sensitivities
  if (is.list(object)) {
    # Check if fdata has been passed to the function to calculate sensitivities
    if (is.null(fdata)) {
      stop("Must be passed fdata to calculate sensitivities of the model")
    }
    # Obtain raw sensitivities
    rawSens <- NeuralSens::SensAnalysisMLP(object,
                                           trData = fdata,
                                           .rawSens = TRUE, plot = FALSE)
  } else if(is.array(object)){
    # The raw sensitivities has been passed instead of the model
    rawSens <- object
  } else {
    stop(paste0("Class ", class(object)," is not accepted as object"))
  }

  for (out in 1:dim(rawSens)[3]) {
    plotdata <- cbind(date.var,as.data.frame(rawSens[,,out]))
    plotdata <- reshape2::melt(plotdata,id.vars = names(plotdata)[1])
    p <- ggplot2::ggplot(plotdata) +
      ggplot2::geom_line(ggplot2::aes(x=plotdata$date.var, y = plotdata$value,
                                      group = plotdata$variable, color = plotdata$variable))

    # See if the user want it faceted
    if (facet) p <- p + ggplot2::facet_grid(plotdata$variable~.)

    return(p)
  }
}
