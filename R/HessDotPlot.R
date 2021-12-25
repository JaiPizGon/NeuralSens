#' Second derivatives 3D scatter or surface plot against input values
#'
#' @description 3D Plot of second derivatives of the neural network output respect
#' to the inputs. This function use \code{plotly} instead of \code{ggplot2} to
#' achieve better visualization
#' @param object fitted neural network model or \code{array} containing the raw
#' second derivatives from the function \code{\link[NeuralSens]{HessianMLP}}
#' @param fdata \code{data.frame} containing the data to evaluate the second derivatives of the model.
#' @param output_vars \code{character vector} with the variables to create the scatter plot. If \code{"all"},
#' then scatter plots are created for all the output variables in \code{fdata}.
#' @param input_vars \code{character vector} with the variables to create the scatter plot in x-axis. If \code{"all"},
#' then scatter plots are created for all the input variables in \code{fdata}.
#' @param input_vars2 \code{character vector} with the variables to create the scatter plot in y-axis. If \code{"all"},
#' then scatter plots are created for all the input variables in \code{fdata}.
#' @param surface \code{logical} if \code{TRUE}, a 3D surface is created instead of 3D scatter plot
#' (only for combinations of different inputs)
#' @param grid \code{logical}. If \code{TRUE}, plots created are show together using \code{\link[gridExtra]{arrangeGrob}}.
#' It does not work on Windows platforms due to bugs in \code{plotly} library.
#' @param color \code{character} specifying the name of a \code{numeric} variable of \code{fdata} to color the 3D scatter plot.
#' @param ... further arguments that should be passed to  \code{\link[NeuralSens]{HessianMLP}} function
#' @return list of 3D \code{geom_point} plots for the inputs variables representing the
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
#' # Try HessDotPlot
#' NeuralSens::HessDotPlot(nnetmod, fdata = nntrData, surface = TRUE, color = "WD")
#' @importFrom magrittr '%>%'
#' @export HessDotPlot
HessDotPlot <- function(object, fdata = NULL, input_vars = "all",
                        input_vars2 = "all", output_vars = "all",
                        surface = FALSE, grid = FALSE, color = NULL, ...) {
  # Check if user wants a grid and is on Windows platform
  if (grid && Sys.info()['sysname'] == "Windows") {
    stop("grid parameter does not work on Windows platforms due to bugs in plotly library.
         This parameter is included for Mac and Linux users.")
  }
  # Check if the object passed is a model or the sensitivities
  if (!is.HessMLP(object)) {
    # Check if fdata has been passed to the function to calculate sensitivities
    if (is.null(fdata)) {
      stop("Must be passed fdata to calculate sensitivities of the model")
    }
    # Obtain raw sensitivities
    HessMLP <- NeuralSens::HessianMLP(object,
                                      trData = fdata,
                                      plot = FALSE,
                                      ...)


  } else if(is.HessMLP(object)){
    # The raw sensitivities has been passed instead of the model
    HessMLP <- object
  } else {
    stop(paste0("Class ", class(object)," is not accepted as object"))
  }

  # Check which plots should be created
  if (output_vars == "all") {
    output_vars <- names(HessMLP$raw_sens)
  }
  if (input_vars == "all") {
    input_vars <- HessMLP$coefnames
  }
  if (input_vars2 == "all") {
    input_vars2 <- HessMLP$coefnames
  }

  if (requireNamespace("plotly") && requireNamespace("magrittr")) {
    plot_for_output <- function(rawSens, fdata, out, inp, inp2, surface, color) {
      plotdata <- as.data.frame(cbind(fdata[,inp], fdata[,inp2], rawSens[inp, inp2,]))
      names(plotdata) <- c("A", "B", "C")
      if (is.null(color)) {
        plotdata$color <- "blue"
      } else {
        plotdata$color <- fdata[,color]
      }
      output_label <- paste0("<sup>",intToUtf8(0x2202L),"<sub>", out,
                             "</sub>", "</sup>", intToUtf8(0x2044L), "<sub>",
                             intToUtf8(0x2202L), "<sub>", inp, "</sub>",
                             intToUtf8(0x2202L),"<sub>", inp2, "</sub>",
                             "</sub>")
      if (surface && inp != inp2) {
        p <- invisible(plotly::plot_ly(plotdata, x = ~A, y = ~B, z = ~C,
                                       type = 'mesh3d', opacity=0.7,
                                       hovertemplate = paste('<i>',inp,'</i>: %{x}<br>',
                                                             '<i>',inp2,'</i>: %{y}<br>',
                                                             '<i>',output_label,'</i>: %{z}<br>'),
                                       scene = paste0(inp, "-", inp2)) %>%
          plotly::layout(scene = list(
            camera = list(
              eye = list(x = min(plotdata[,"A"]), y = max(plotdata[,"B"]), z = 0)
            ),
            xaxis = list(title = inp),
            yaxis = list(title = inp2),
            zaxis = list(title = output_label))))

      } else {
        p <- invisible(plotly::plot_ly(plotdata, x = ~A, y = ~B, z = ~C, color = ~color,
                                       scene = paste0(inp, "-", inp2)) %>%
          plotly::layout(scene = list(
            camera = list(
              eye = list(x = min(plotdata[,"A"]), y = max(plotdata[,"B"]), z = 0)
            ),
            xaxis = list(title = inp),
            yaxis = list(title = inp2),
            zaxis = list(title = output_label))) %>%
          plotly::add_markers(opacity = 0.7, marker=list(line=list(color='black', width=1)),
                              hovertemplate = paste('<i>',inp,'</i>: %{x}<br>',
                                                    '<i>',inp2,'</i>: %{y}<br>',
                                                    '<i>',output_label,'</i>: %{z}<br>',
                                                    '<i>',color,'</i>: %{marker.color}'))) %>%
          plotly::colorbar(title = color)
      }
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
      gr2 <- list()
      input_vars_comb <- expand.grid(input_vars, input_vars2)
      index_comb <- 1:length(unique(c(input_vars, input_vars2)))
      names(index_comb) <- unique(c(input_vars, input_vars2))
      num_comb <- apply(input_vars_comb, 2, function(x) index_comb[x])
      num_comb <- cbind(num_comb, num_comb[,1] + num_comb[,2])
      input_vars_comb <- input_vars_comb[!duplicated(num_comb[,3]),]
      input_vars_comb[,"comb"] <- paste(input_vars_comb[,1],input_vars_comb[,2], sep = "-")
      for (irow in 1:nrow(input_vars_comb)) {
        plotlist[[out]][[input_vars_comb[irow,"comb"]]] <- plot_for_output(HessMLP$raw_sens[[out]],
                                                                          as.data.frame(HessMLP$trData),
                                                                          out, as.character(input_vars_comb[irow,1]),
                                                                          as.character(input_vars_comb[irow,2]), surface, color)
      }
      if (grid) {
        gr[[out]] <- plotly::subplot(plotlist[[out]], nrows=length(plotlist[[out]]), titleY=T, shareY = TRUE)
      }
    }
    if (grid) {
      plotly::subplot(gr2, nrows=1, titleY=T, shareX = TRUE)
    }
    return(invisible(plotlist))
  }
}
