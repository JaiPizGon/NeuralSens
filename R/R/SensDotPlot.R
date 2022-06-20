#' Sensitivity scatter plot against input values
#'
#' @description Plot of sensitivities of the neural network output respect
#' to the inputs
#' @param object fitted neural network model or \code{array} containing the raw
#' sensitivities from the function \code{\link[NeuralSens]{SensAnalysisMLP}}
#' @param fdata \code{data.frame} containing the data to evaluate the sensitivity of the model.
#' @param output_vars \code{character vector} with the variables to create the scatter plot. If \code{"all"},
#' then scatter plots are created for all the output variables in \code{fdata}.
#' @param input_vars \code{character vector} with the variables to create the scatter plot. If \code{"all"},
#' then scatter plots are created for all the input variables in \code{fdata}.
#' @param smooth \code{logical} if \code{TRUE}, \code{geom_smooth} plots are added to each variable plot
#' @param nspline \code{integer} if \code{smooth} is TRUE, this determine the degree of the spline used
#' to perform \code{geom_smooth}. If \code{nspline} is NULL, the square root of the length of the data
#' is used as degrees of the spline.
#' @param grid \code{logical}. If \code{TRUE}, plots created are show together using \code{\link[gridExtra]{arrangeGrob}}
#' @param color \code{character} specifying the name of a \code{numeric} variable of \code{fdata} to color the scatter plot.
#' @param ... further arguments that should be passed to  \code{\link[NeuralSens]{SensAnalysisMLP}} function
#' @return list of \code{geom_point} plots for the inputs variables representing the
#' sensitivity of each output respect to the inputs
#' @examples
#' ## Load data -------------------------------------------------------------------
#' data("DAILY_DEMAND_TR")
#' fdata <- DAILY_DEMAND_TR
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
#' # Try SensDotPlot
#' NeuralSens::SensDotPlot(nnetmod, fdata = nntrData)
#' @importFrom magrittr '%>%'
#' @export SensDotPlot
SensDotPlot <- function(object, fdata = NULL, input_vars = "all",
                        output_vars = "all", smooth = FALSE,
                        nspline = NULL, color = NULL, grid = FALSE, ...) {
  if (is.HessMLP(object)) {
    object <- HessToSensMLP(object)
  }
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


  } else if(is.SensMLP(object)){
    # The raw sensitivities has been passed instead of the model
    SensMLP <- object
  } else {
    stop(paste0("Class ", class(object)," is not accepted as object"))
  }

  # Check which plots should be created
  if (output_vars == "all") {
    output_vars <- names(SensMLP$raw_sens)
  }
  if (input_vars == "all") {
    input_vars <- SensMLP$coefnames
  }
  # Check degree of spline
  if (is.null(nspline)) {
    nspline <- floor(sqrt(dim(SensMLP$raw_sens[[1]])[1]))
  }
  plot_for_output <- function(rawSens, fdata, out, inp, smooth, color) {
    plotdata <- as.data.frame(cbind(fdata[,inp], rawSens[,inp]))
    if (is.null(color)) {
      plotdata[,'color'] <- 'blue'
    } else {
      plotdata[,'color'] <- plotdata[,color]
    }
    p <- ggplot2::ggplot(plotdata, ggplot2::aes(x = plotdata[,1],
                                                y = plotdata[,2],
                                                color = plotdata[,'color'])) +
      ggplot2::geom_point() +
      ggplot2::xlab(inp) +
      ggplot2::ylab(as.expression(bquote(partialdiff~.(out)~"/"~partialdiff~.(inp))))
    # See if the user want a smooth plot
    if (smooth) p <- p + ggplot2::geom_smooth(method = "lm", color = "blue", formula = y ~ splines::bs(x, nspline), se = FALSE)
    # See if the user want it faceted
    if (!grid) {
      print(p)
    }
    return(p)
  }
  plotlist <- list()
  gr <- list()
  for (out in output_vars) {
    plotlist[[out]] <- list()
    for (inp in input_vars) {
      plotlist[[out]][[inp]] <- plot_for_output(SensMLP$raw_sens[[out]],
                                                as.data.frame(SensMLP$trData),
                                                out, inp, smooth, color)
    }
    if (grid) {
      gr[[out]] <- gridExtra::arrangeGrob(grobs = plotlist[[out]], ncol = 1, top=out)
    }
  }
  if (grid) {
    gridExtra::grid.arrange(grobs=gr, ncol=length(output_vars))
  }
  return(invisible(plotlist))
}
