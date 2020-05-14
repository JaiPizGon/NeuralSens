#' Plot sensitivities of a neural network model
#'
#' @description Function to plot the sensitivities created by \code{\link[NeuralSens]{HessianMLP}}.
#' @param hess \code{HessMLP} object created by \code{\link[NeuralSens]{HessianMLP}}.
#' @param sens \code{SensMLP} object created by \code{\link[NeuralSens]{SensAnalysisMLP}}.
#' @param output \code{numeric} or {character} specifying the output neuron or output name to be plotted.
#' By default is the first output (\code{output = 1}).
#' @param senstype \code{character} specifying the type of plot to be plotted. It can be "matrix" or
#'  "interactions". If type = "matrix", only the second derivatives are plotted. If type = "interactions"
#'  the main diagonal are the first derivatives respect each input variable.
#' @param metric \code{character} specifying the metric to be plotted. It can be "mean",
#' "std" or "meanSensSQ".
#' @param ... further argument passed similar to \code{ggcorrplot} arguments.
#' @return a list of \code{\link[ggplot2]{ggplot}}s, one for each output neuron.
#' @details Most of the code of this function is based on
#' \code{ggcorrplot()} function from package \code{ggcorrplot}. However, due to the
#' inhability of changing the limits of the color scale, it keeps giving a warning
#' if that function is used and the color scale overwritten.
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
#' # Try HessianMLP
#' H <- NeuralSens::HessianMLP(nnetmod, trData = nntrData, plot = FALSE)
#' NeuralSens::SensMatPlot(H)
#' S <- NeuralSens::SensAnalysisMLP(nnetmod, trData = nntrData, plot = FALSE)
#' NeuralSens::SensMatPlot(H, S, senstype = "interactions")
#' @export SensMatPlot
SensMatPlot <- function(hess, sens = NULL, output = 1, metric = c("mean","std","meanSensSQ"),
                        senstype = c("matrix","interactions"), ...) {
  ggMatPlot <- function(corr, colors = c("blue", "white", "red"),
                        senstype = "matrix",
                        method = c("square", "circle"),
                        type = c("full", "lower", "upper"),
                        ggtheme = ggplot2::theme_minimal, title = "",
                        show.legend = TRUE, show.diag = FALSE,
                        outline.color = "gray", hc.order = FALSE,
                        hc.method = "complete", lab = FALSE,
                        lab_col = "black", lab_size = 4,
                        p.mat = NULL, sig.level = 0.05, insig = c("pch", "blank"),
                        pch = 4, pch.col = "black", pch.cex = 5,
                        tl.cex = 12, tl.col = "black", tl.srt = 45, digits = 2,
                        as.is = FALSE, ...) {
    ## Helper functions
    .get_lower_tri <- function(cormat, show.diag = FALSE) {
      if (is.null(cormat)) {
        return(cormat)
      }
      cormat[upper.tri(cormat)] <- NA
      if (!show.diag) {
        diag(cormat) <- NA
      }
      return(cormat)
    }

    # Get upper triangle of the correlation matrix
    .get_upper_tri <- function(cormat, show.diag = FALSE) {
      if (is.null(cormat)) {
        return(cormat)
      }
      cormat[lower.tri(cormat)] <- NA
      if (!show.diag) {
        diag(cormat) <- NA
      }
      return(cormat)
    }

    # hc.order correlation matrix
    .hc_cormat_order <- function(cormat, hc.method = "complete") {
      dd <- stats::as.dist((1 - cormat) / 2)
      hc <- stats::hclust(dd, method = hc.method)
      hc$order
    }

    .no_panel <- function() {
      ggplot2::theme(
        axis.title.x = ggplot2::element_blank(),
        axis.title.y = ggplot2::element_blank()
      )
    }


    # Convert a tbl to matrix
    .tibble_to_matrix <- function(x){
      x <-  as.data.frame(x)
      rownames(x) <- x[, 1]
      x <- x[, -1]
      as.matrix(x)
    }

    ## GGCORRPLOT function
    type <- match.arg(type)
    method <- match.arg(method)
    insig <- match.arg(insig)

    corr <- as.matrix(corr)

    corr <- base::round(x = corr, digits = digits)

    if (hc.order) {
      ord <- .hc_cormat_order(corr)
      corr <- corr[ord, ord]
      if (!is.null(p.mat)) {
        p.mat <- p.mat[ord, ord]
        p.mat <- base::round(x = p.mat, digits = digits)
      }
    }

    # Get lower or upper triangle
    if (type == "lower") {
      corr <- .get_lower_tri(corr, show.diag)
      p.mat <- .get_lower_tri(p.mat, show.diag)
    }
    else if (type == "upper") {
      corr <- .get_upper_tri(corr, show.diag)
      p.mat <- .get_upper_tri(p.mat, show.diag)
    }

    # Melt corr and pmat
    corr <- reshape2::melt(corr, na.rm = TRUE, as.is = as.is)
    colnames(corr) <- c("Var1", "Var2", "value")
    corr$pvalue <- rep(NA, nrow(corr))
    corr$signif <- rep(NA, nrow(corr))

    if (!is.null(p.mat)) {
      p.mat <- reshape2::melt(p.mat, na.rm = TRUE)
      corr$coef <- corr$value
      corr$pvalue <- p.mat$value
      corr$signif <- as.numeric(p.mat$value <= sig.level)
      p.mat <- subset(p.mat, p.mat$value > sig.level)
      if (insig == "blank") {
        corr$value <- corr$value * corr$signif
      }
    }


    corr$abs_corr <- abs(corr$value) * 10
    corr$value_diag <- corr$value
    corr$value_diag[corr$Var1 != corr$Var2] <- NA
    corr$value_inter <- corr$value
    corr$value_inter[corr$Var1 == corr$Var2] <- NA
    # heatmap
    p <-
      ggplot2::ggplot(
        data = corr,
        mapping = ggplot2::aes_string(x = "Var1", y = "Var2")
      )

    if(senstype == "matrix") {
      # modification based on method
      if (method == "square") {
        p <- p +
          ggplot2::geom_tile(ggplot2::aes_string(fill = "value"),color = outline.color)
      } else if (method == "circle") {
        p <- p +
          ggplot2::geom_point(
            color = outline.color,
            shape = 21,
            ggplot2::aes_string(fill = "value",size = "abs_corr")
          ) +
          ggplot2::scale_size(range = c(4, 10)) +
          ggplot2::guides(size = FALSE)
      }
      # adding colors
      p <-
        p + ggplot2::scale_fill_gradient2(
          low = colors[1],
          high = colors[3],
          mid = colors[2],
          midpoint = 0,
          space = "Lab",
          limits = c(-1,1)*max(corr$value),
          name = "Second derivatives"
        )
    } else {
      # modification based on method
      if (method == "square") {
        p <- p +
          ggplot2::geom_tile(ggplot2::aes_string(fill = "value_inter"),color = outline.color) +
          ggplot2::scale_fill_gradient2(
            low = colors[3], high = colors[1],
            mid = colors[2], space = "Lab",
            limits = c(-1,1)*max(corr$value_inter),
            name = "Interactions", na.value = "transparent") +
          ggnewscale::new_scale("fill") +
          ggplot2::geom_tile(ggplot2::aes_string(fill = "value_diag"),color = outline.color) +
          ggplot2::scale_fill_gradient2(
            low = colors[3], high = colors[1],
            mid = colors[2], space = "Lab",
            limits = c(-1,1)*max(corr$value_diag),
            name = "Sensitivities", na.value = "transparent")
      } else if (method == "circle") {
        p <- p +
          ggplot2::geom_point(ggplot2::aes_string(fill = "value_inter",size = "abs_corr"),
                              color = outline.color,
                              shape = 21
          ) +
          ggplot2::scale_size(range = c(4, 10), guide = 'none') +
          ggplot2::scale_fill_gradient2(
            low = colors[3], high = colors[1],
            mid = colors[2], space = "Lab",
            limits = c(-1,1)*max(corr$value_inter),
            name = "Interactions", na.value = "transparent") +
          ggnewscale::new_scale("fill") +
          ggnewscale::new_scale("size") +
          ggplot2::geom_point(ggplot2::aes_string(fill = "value_diag",size = "abs_corr"),
                              color = outline.color,
                              shape = 21
          ) +
          ggplot2::scale_size(range = c(4, 10),guide = 'none') +
          ggplot2::scale_fill_gradient2(
            low = colors[3], high = colors[1],
            mid = colors[2], space = "Lab",
            limits = c(-1,1)*max(corr$value_diag),
            name = "Sensitivities", na.value = "transparent")
      }
    }


    # depending on the class of the object, add the specified theme
    if (class(ggtheme)[[1]] == "function") {
      p <- p + ggtheme()
    } else if (class(ggtheme)[[1]] == "theme") {
      p <- p + ggtheme
    }

    p <- p +
      ggplot2::theme(
        axis.text.x = ggplot2::element_text(
          angle = tl.srt,
          vjust = 1,
          size = tl.cex,
          hjust = 1
        ),
        axis.text.y = ggplot2::element_text(size = tl.cex)
      ) +
      ggplot2::coord_fixed()

    label <- round(x = corr[, "value"], digits = digits)
    if(!is.null(p.mat) & insig == "blank"){
      ns <- corr$pvalue > sig.level
      if(sum(ns) > 0) label[ns] <- " "
    }

    # matrix cell labels
    if (lab) {
      p <- p +
        ggplot2::geom_text(
          mapping = ggplot2::aes_string(x = "Var1", y = "Var2"),
          label = label,
          color = lab_col,
          size = lab_size
        )
    }

    # matrix cell glyphs
    if (!is.null(p.mat) & insig == "pch") {
      p <- p + ggplot2::geom_point(
        data = p.mat,
        mapping = ggplot2::aes_string(x = "Var1", y = "Var2"),
        shape = pch,
        size = pch.cex,
        color = pch.col
      )
    }

    # add titles
    if (title != "") {
      p <- p +
        ggplot2::ggtitle(title)
    }

    # removing legend
    if (!show.legend) {
      p <- p +
        ggplot2::theme(legend.position = "none")
    }

    # removing panel
    p <- p +
      .no_panel()
    p
  }
  metric <- match.arg(metric)
  senstype <- match.arg(senstype)
  if (!is.HessMLP(hess)) {
    stop("Function only works with HessMLP objects")
  }
  if ((is.null(sens) || !is.SensMLP(sens) )&& senstype == "interactions") {
    stop("If senstype == 'interactions', sensitivities must be provided in sens argument")
  }
  plotlist <- list()
  if (senstype == "matrix") {
    for (out in 1:length(hess$sens)) {
      plotlist[[out]] <- list()
      for (met in 1:length(hess$sens[[out]])) {

          plotlist[[out]][[met]] <- ggMatPlot(hess$sens[[out]][[met]], senstype = senstype, ...) +
            ggplot2::ggtitle(paste0("Matrix plot of second derivatives of metric ",names(hess$sens[[out]])[met],
                                    " of output ",names(hess$sens)[out]))
      }
      names(plotlist[[out]]) <- names(hess$sens[[out]])
    }
    names(plotlist) <- names(hess$sens)
  } else {
    for (out in 1:length(hess$sens)) {
      plotlist[[out]] <- list()
      for (met in 1:length(hess$sens[[out]])) {
        ders <- hess$sens[[out]][[met]]
        diag(ders) <- sens$sens[[out]][[met]]
        plotlist[[out]][[met]] <- ggMatPlot(ders, senstype = senstype, ...) +
          ggplot2::ggtitle(paste0("Interactions plot of metric ",names(hess$sens[[out]])[met],
                                  " of output ",names(hess$sens)[out]))
      }
      names(plotlist[[out]]) <- names(hess$sens[[out]])
    }
    names(plotlist) <- names(hess$sens)
  }
  graphics::plot(plotlist[[output]][[metric]])
  return(invisible(plotlist))
}
