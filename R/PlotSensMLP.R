#' Neural network structure sensitivity plot
#'
#' @description Plot a neural interpretation diagram colored by sensitivities
#' of the model
#' @param MLP.fit fitted neural network model
#' @param metric metric to plot in the NID. It can be "mean" (default) or "sqmean".
#' It can be any metric to combine the raw sensitivities
#' @param sens_neg_col \code{character} string indicating color of negative sensitivity
#'  measure, default 'red'. The same is passed to argument \code{neg_col} of
#'  \link[NeuralNetTools:plotnet]{plotnet}
#' @param sens_pos_col \code{character} string indicating color of positive sensitivity
#'  measure, default 'blue'. The same is passed to argument \code{pos_col} of
#'  \link[NeuralNetTools:plotnet]{plotnet}
#' @param ...	additional arguments passed to \link[NeuralNetTools:plotnet]{plotnet} and/or
#' \link[NeuralSens:SensAnalysisMLP]{SensAnalysisMLP}
#' @return A graphics object
#' @examples
#' ## Load data -------------------------------------------------------------------
#' data("DAILY_DEMAND_TR")
#' fdata <- DAILY_DEMAND_TR
#' ## Parameters of the NNET ------------------------------------------------------
#' hidden_neurons <- 5
#' iters <- 100
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
#' # Try SensAnalysisMLP
#' NeuralSens::PlotSensMLP(nnetmod, trData = nntrData)
#' @export PlotSensMLP
PlotSensMLP <- function(MLP.fit, metric = "mean",
                        sens_neg_col = "red", sens_pos_col = "blue",
                        ...) {

  # First obtain all derivatives of the model
  Derivatives <- SensAnalysisMLP(MLP.fit, plot = FALSE, ..., return_all_sens = TRUE)
  # Pull apart derivatives of layers and weights
  d <- Derivatives[[1]]
  mlpstr <- Derivatives[[2]]
  wts <- Derivatives[[3]]
  # Stored the length of colors needed
  color_lengths <- sapply(d, function(x){dim(x)[1]})
  sens <- list()
  for (i in 1:length(color_lengths)) {
    der <- aperm(d[[i]], c(3,1,2))
    der <- CombineSens(der, metric)
    # Apply metric to calculate
    if(is.function(metric)) {
      sens[[i]] <- apply(der, 2, metric)
    } else if (metric == "mean") {
      sens[[i]] <- apply(der, 2, mean, na.rm = TRUE)
    } else if (metric == "sqmean") {
      sens[[i]] <- apply(der, 2, function(x){mean(x^2, na.rm = TRUE)})
    } else {
      stop("metric must be a function to calculate over a column of sensitivities")
    }
  }
  # Collapse all sensitivities
  sens <- do.call("c",sens)

  # Rescale the sensitivities in order to obtain the colors
  sens_scaled <- sign(sens) * round(scales::rescale(abs(sens),c(1,ceiling(1/min(sens))))) + ceiling(1/min(sens)) + 1

  colPal <- grDevices::colorRampPalette(c(sens_neg_col, "white", sens_pos_col))
  senscolors <- colPal(max(sens_scaled) + 1)[round(sens_scaled)]
  senscolors_list <- list()
  color_lengths <- c(0,color_lengths)
  for (i in 2:length(color_lengths)) {
    senscolors_list[[i-1]] <- senscolors[(cumsum(color_lengths)[i-1]+1):cumsum(color_lengths)[i]]
  }

  # Plot Neural network NID and legend scale
  graphics::layout(matrix(1:2,nrow =1), widths = c(0.8,0.2))
  op <- graphics::par(mar=c(5.1,1.1,4.1,2.1))
  NeuralNetTools::plotnet(wts, mlpstr,
                          circle_col = senscolors_list,
                          pos_col = sens_pos_col,
                          neg_col = sens_neg_col,
                          bord_col = "black",
                          bias = FALSE,
                          x_names = Derivatives[[4]],
                          y_names = Derivatives[[5]],
                          ...)
  # Substitute the output points to avoid changing its colors
  args <- list(...)
  out_pos <- 0.4
  if ("pad_x" %in% names(args)) {
    out_pos <- args$pad_x * 0.4
  }
  # Default values for arguments
  cex_val <- 1
  circle_cex <- 5
  node_labs <- TRUE
  bord_col <- "black"
  in_col <- "white"
  line_stag <- NULL
  var_labs <- TRUE
  max_sp <- FALSE
  layer_points_args <- c("cex_val", "circle_cex", "bord_col","node_labs","line_stag",
                         "var_labs","max_sp")
  # Check if any argument must be changed
  for (i in which(layer_points_args %in% names(args))) {
    eval(paste0(layer_points_args[i]," <- args$",layer_points_args[i]))
  }
  x_range <- c(-1,1)
  y_range <- c(0,1)
  # Substitute output values
  layer <- mlpstr[length(mlpstr)]

  x <- rep(out_pos * diff(x_range), layer)
  if(max_sp){
    spacing <- diff(c(0 * diff(y_range), 0.9 * diff(y_range)))/layer
  } else {
    spacing <- diff(c(0 * diff(y_range), 0.9 * diff(y_range)))/max(mlpstr)
  }

  y <- seq(0.5 * (diff(y_range) + spacing * (layer - 1)), 0.5 * (diff(y_range) - spacing * (layer - 1)),
             length = layer)

  graphics::points(x, y, pch = 21, cex = circle_cex, col = bord_col, bg = in_col)
  if(node_labs) graphics::text(x, y, paste('O', 1:layer, sep = ''), cex = cex_val)
  graphics::text(x + line_stag * diff(x_range), y,  Derivatives[[5]], pos = 4, cex = cex_val)

  # Create color scale to know the sensitivities in the graph
  xl <- 1
  yb <- 1
  xr <- 1.5
  yt <- 2

  graphics::par(mar=c(5.1,0.5,4.1,0.5))
  needed_colors <- sum(mlpstr[1:(length(mlpstr)-1)])
  graphics::plot(NA,type="n",ann=FALSE,xlim=c(1,2),ylim=c(1,2),xaxt="n",yaxt="n",bty="n")
  graphics::rect(
    xl,
    utils::head(seq(yb,yt,(yt-yb)/needed_colors),-1),
    xr,
    utils::tail(seq(yb,yt,(yt-yb)/needed_colors),-1),
    col=stats::na.omit(do.call("c",senscolors_list[1:(length(mlpstr)-1)])[order(sens[1:needed_colors])])
  )
  graphics::text(x = 1.25, y = rowMeans(cbind(utils::head(seq(yb,yt,(yt-yb)/needed_colors),-1),
                                              utils::tail(seq(yb,yt,(yt-yb)/needed_colors),-1))),
       label = c(paste0("I",1:mlpstr[1]),paste0("H",1:sum(mlpstr[2:(length(mlpstr)-1)])))[order(sens[1:needed_colors])])
  graphics::mtext(round(sens[1:needed_colors], digits = ifelse(any(abs(sens[1:needed_colors]) < 1),-floor(log10(min(sens[1:needed_colors]))),2))[order(sens[1:needed_colors])],
                  side=2, at=utils::tail(seq(yb,yt,(yt-yb)/sum(mlpstr[1:(length(mlpstr)-1)])),-1)-0.05,
                  las=2,cex=0.7)
  reset.graphics <- function(oldpar) {
    graphics::par(oldpar)
    graphics::layout(matrix(1,nrow =1), widths = 1)
  }
  on.exit(reset.graphics(op))
}

