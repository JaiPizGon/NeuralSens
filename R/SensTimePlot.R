#' Sensitivity analysis plot over time of the data
#'
#' @description Plot of sensitivity of the neural network output respect
#' to the inputs over the time variable from the data provided
#' @param object fitted neural network model or \code{array} containing the raw
#' sensitivities from the function \code{\link[NeuralSens]{SensAnalysisMLP}}
#' @param fdata \code{data.frame} containing the data to evaluate the sensitivity of the model.
#' Not needed if the raw sensitivities has been passed as \code{object}
#' @param date.var \code{Posixct vector} with the date of each sample of \code{fdata}
#' If \code{NULL}, the first variable with Posixct format of \code{fdata} is used as dates
#' @param facet \code{logical} if \code{TRUE}, function \code{facet_grid} from \code{ggplot2} is used
#' @param smooth \code{logical} if \code{TRUE}, \code{geom_smooth} plots are added to each variable plot
#' @param nspline \code{integer} if \code{smooth} is TRUE, this determine the degree of the spline used
#' to perform \code{geom_smooth}. If \code{nspline} is NULL, the square root of the length of the timeseries
#' is used as degrees of the spline.
#' @param ... further arguments that should be passed to  \code{\link[NeuralSens]{SensAnalysisMLP}} function
#' @return list of \code{geom_line} plots for the inputs variables representing the
#' sensitivity of each output respect to the inputs over time
#' @examples
#' ## Load data -------------------------------------------------------------------
#' data("DAILY_DEMAND_TR")
#' fdata <- DAILY_DEMAND_TR
#' fdata[,3] <- ifelse(as.data.frame(fdata)[,3] %in% c("SUN","SAT"), 0, 1)
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
#'                       data = nntrData,
#'                       linear.output = TRUE,
#'                       size = hidden_neurons,
#'                       decay = decay,
#'                       maxit = iters)
#' # Try SensTimePlot
#' NeuralSens::SensTimePlot(nnetmod, fdata = nntrData, date.var = NULL)
#' @export SensTimePlot
SensTimePlot <- function(object, fdata = NULL, date.var = NULL, facet = FALSE,
                         smooth = FALSE, nspline = NULL, ...) {
  # Check if the object passed is a model (list) or the raw sensitivities
  if (is.list(object)) {
    # Check if fdata has been passed to the function to calculate sensitivities
    if (is.null(fdata)) {
      stop("Must be passed fdata to calculate sensitivities of the model")
    }
    # Obtain raw sensitivities
    rawSens <- NeuralSens::SensAnalysisMLP(object,
                                           trData = fdata,
                                           .rawSens = TRUE,
                                           plot = FALSE, ...)

  } else if(is.array(object)){
    # The raw sensitivities has been passed instead of the model
    rawSens <- object
  } else {
    stop(paste0("Class ", class(object)," is not accepted as object"))
  }
  # Check if the variable name of the date has been specified
  if (is.null(date.var)) {
    if (any(apply(fdata, 2, function(x){inherits(x,"POSIXct") || inherits(x,"POSIXlt")}))) {
      date.var <- fdata[,sapply(fdata, function(x){
        inherits(x,"POSIXct") || inherits(x,"POSIXlt")})]
    } else {
      date.var <- seq_len(dim(rawSens)[1])
    }
  }
  # Check degree of spline
  if (is.null(nspline)) {
    nspline <- floor(sqrt(dim(rawSens)[1]))
  }
  plotlist <- list()
  for (out in 1:dim(rawSens)[3]) {
    local({
      out <- out
      plotdata <- cbind(date.var,as.data.frame(rawSens[,,out]))
      plotdata <- reshape2::melt(plotdata,id.vars = names(plotdata)[1])
      p <- ggplot2::ggplot(plotdata, ggplot2::aes(x = plotdata$date.var, y = plotdata$value,
                                                  group = plotdata$variable, color = plotdata$variable)) +
        ggplot2::geom_line() +
        ggplot2::labs(color = "Inputs") +
        ggplot2::xlab("Time") +
        ggplot2::ylab(NULL)
      # See if the user want a smooth plot
      if (smooth) p <- p + ggplot2::geom_smooth(method = "lm", color = "blue", formula = y ~ splines::bs(x, nspline), se = FALSE)
      # See if the user want it faceted
      if (facet) p <- p + ggplot2::facet_grid(plotdata$variable~., scales = "free_y")
      print(p)
      plotlist[[out]] <<- p
    })
  }
  return(invisible(plotlist))
}
