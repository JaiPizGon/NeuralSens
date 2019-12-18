#' Plot sensitivities of a neural network model
#'
#' @description Function to plot the sensitivities created by \code{\link[NeuralSens]{SensAnalysisMLP}}.
#' @param sens \code{data.frame} with the sensitivities calculated by \code{\link[NeuralSens]{SensAnalysisMLP}} using \code{.rawSens = FALSE}.
#' @param der \code{matrix} with the sensitivities calculated by \code{\link[NeuralSens]{SensAnalysisMLP}} using \code{.rawSens = TRUE}.
#' @return Plots: \itemize{ \item Plot 1: colorful plot with the
#'   classification of the classes in a 2D map \item Plot 2: b/w plot with
#'   probability of the chosen class in a 2D map \item Plot 3: plot with the
#'   stats::predictions of the data provided if param \code{dens} is not \code{NULL}}
#' @details Due to the fact that \code{sens} is calculated from \code{dens}, if the latter is passed as argument
#' the argument \code{sens} is overwritten to maintain coherence between the three plots even. If only \code{sens} is
#' given, the last plot with the density plots of the inputs is not calculated.
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
#' NeuralSens::SensitivityPlots(sens)
#' sensraw <- NeuralSens::SensAnalysisMLP(nnetmod, trData = nntrData, plot = FALSE, .rawSens = TRUE)
#' NeuralSens::SensitivityPlots(der = sensraw[,,1])
#' @export SensitivityPlots
SensitivityPlots <- function(sens = NULL,der = NULL) {
  plotlist <- list()

  # Check that at least the one argument has been passed
  # If only der is passed, sens is calculated as in SensAnalysisMLP

  if(!is.null(der)) {
    # Calculate sensitivities if der is passed and overwrite sens
    sens <-
      data.frame(
        varNames = colnames(der),
        mean = colMeans(der, na.rm = TRUE),
        std = apply(der, 2, stats::sd, na.rm = TRUE),
        meanSensSQ = colMeans(der ^ 2, na.rm = TRUE)
      )
    # Don't know why the names are overwritten so must be written again
    names(sens) <- c("varNames", "mean", "std", "meanSensSQ")
  } else if(is.null(sens)) {
    stop("Sensitivities must be passed to the function, use NeuralSens::SensAnalysisMLP to calculate them")
  }
  # Order sensitivity measures by importance order
  sens <- sens[order(sens$meanSensSQ),]
  sens$varNames <- factor(sens$varNames, levels = sens$varNames[order(sens$meanSensSQ)])

  plotlist[[1]] <- ggplot2::ggplot(sens) +
    ggplot2::geom_point(ggplot2::aes(x = 0, y = 0), size = 5, color = "blue") +
    ggplot2::geom_hline(ggplot2::aes(yintercept = 0), color = "blue") +
    ggplot2::geom_vline(ggplot2::aes(xintercept = 0), color = "blue") +
    ggplot2::geom_point(ggplot2::aes_string(x = "mean", y = "std")) +
    ggplot2::geom_label(ggplot2::aes_string(x = "mean", y = "std", label = "varNames"),
                        position = "nudge") +
    # coord_cartesian(xlim = c(min(sens$mean,0)-0.1*abs(min(sens$mean,0)), max(sens$mean)+0.1*abs(max(sens$mean))), ylim = c(0, max(sens$std)*1.1))+
    ggplot2::labs(x = "mean(Sens)", y = "std(Sens)")


  plotlist[[2]] <- ggplot2::ggplot(sens) +
    ggplot2::geom_col(ggplot2::aes_string(x = "varNames", y = "meanSensSQ", fill = "meanSensSQ")) +
    ggplot2::labs(x = "Input variables", y = "mean(Sens^2)") + ggplot2::guides(fill = "none")

  if (!is.null(der)) {
    # If the raw values of the derivatives has been passed to the function
    # the density plots of each of these derivatives can be extracted and plotted
    der2 <- as.data.frame(der)
    names(der2) <- sens$varNames
    # Remove any variable which is all zero -> pruned variable
    der2 <- der2[,!sapply(der2,function(x){all(x ==  0)})]
    dataplot <- reshape2::melt(der2, measure.vars = names(der2))

    # Check the right x limits for the density plots
    quant <- stats::quantile(abs(dataplot$value), c(0.8, 1))
    if (10*quant[1] < quant[2]) { # Distribution has too much dispersion
      xlim <- c(1,-1)*max(abs(stats::quantile(dataplot$value, c(0.2,0.8))))
    } else {
      xlim <- c(-1.1, 1.1)*max(abs(dataplot$value), na.rm = TRUE)
    }

    plotlist[[3]] <- ggplot2::ggplot(dataplot) +
      ggplot2::geom_density(ggplot2::aes_string(x = "value", fill = "variable"),
                            alpha = 0.4,
                            bw = "bcv") +
      ggplot2::labs(x = "Sens", y = "density(Sens)") +
      ggplot2::xlim(xlim)
      # ggplot2::xlim(-2 * max(sens$std, na.rm = TRUE), 2 * max(sens$std, na.rm = TRUE))
  }
  # Plot the list of plots created before
  gridExtra::grid.arrange(grobs = plotlist,
                          nrow  = length(plotlist),
                          ncols = 1)
  # Return the plots created if the user want to edit them by hand
  return(invisible(plotlist))
}
