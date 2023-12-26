#' Plot sensitivities of a neural network model
#'
#' @description Function to plot the sensitivities created by \code{\link[NeuralSens]{SensAnalysisMLP}}.
#' @param sens \code{SensAnalysisMLP} object created by \code{\link[NeuralSens]{SensAnalysisMLP}} or \code{HessMLP} object
#' created by \code{\link[NeuralSens]{HessianMLP}}.
#' @param der \code{logical} indicating if density plots should be created. By default is \code{TRUE}
#' @param zoom \code{logical} indicating if the distributions should be zoomed when there is any of them which is too tiny to be appreciated in the third plot.
#' \code{\link[ggforce]{facet_zoom}} function from \code{ggforce} package is required.
#' @param quit.legend \code{logical} indicating if legend of the third plot should be removed. By default is \code{FALSE}
#' @param output \code{numeric} or \code{character} specifying the output neuron or output name to be plotted.
#' By default is the first output (\code{output = 1}).
#' @param plot_type \code{character} indicating which of the 3 plots to show. Useful when several variables are analyzed.
#' Acceptable values are 'mean_sd', 'square', 'raw' corresponding to first, second and third plot respectively. If \code{NULL},
#' all plots are shown at the same time. By default is \code{NULL}.
#' @param inp_var \code{character} indicating which input variable to show in density plot. Only useful when
#' choosing plot_type='raw' to show the density plot of one input variable. If \code{NULL}, all variables
#' are plotted in density plot. By default is \code{NULL}.
#' @param title \code{character} title of the sensitivity plots
#' @param dodge_var \code{bool} Flag to indicate that x ticks in meanSensSQ plot must dodge between them. Useful with
#' too long input names.
#' @return List with the following plot for each output: \itemize{ \item Plot 1: colorful plot with the
#'   classification of the classes in a 2D map \item Plot 2: b/w plot with
#'   probability of the chosen class in a 2D map \item Plot 3: plot with the
#'   stats::predictions of the data provided if param \code{der} is \code{FALSE}}
#' @references
#' Pizarroso J, Portela J, Muñoz A (2022). NeuralSens: Sensitivity Analysis of
#' Neural Networks. Journal of Statistical Software, 102(7), 1-36.
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
#' @export SensitivityPlots
SensitivityPlots <- function(sens = NULL, der = TRUE,
                             zoom = TRUE, quit.legend = FALSE,
                             output = 1, plot_type=NULL,
                             inp_var=NULL, title='Sensitivity Plots',
                             dodge_var = FALSE) {
  if (is.array(der)) stop("der argument is no more the raw sensitivities due to creation of SensMLP class. Check ?SensitivityPlots for more information")
  if (is.HessMLP(sens)) {
    sens <- HessToSensMLP(sens)
  }
  plotlist <- list()
  sens_orig <- sens
  pl <- list()
  for (out in 1:length(sens_orig$sens)) {
    sens <- sens_orig$sens[[out]]
    raw_sens <- sens_orig$raw_sens[[out]]
    # Order sensitivity measures by importance order
    orig_order <- order(sens$meanSensSQ)
    sens <- sens[orig_order,]
    sens$varNames <- factor(rownames(sens), levels = rownames(sens)[order(sens$meanSensSQ)])

    plotlist[[1]] <- ggplot2::ggplot(sens) +
      ggplot2::geom_point(ggplot2::aes(x = 0, y = 0), size = 5, color = "blue") +
      ggplot2::geom_hline(ggplot2::aes(yintercept = 0), color = "blue") +
      ggplot2::geom_vline(ggplot2::aes(xintercept = 0), color = "blue") +
      ggplot2::geom_point(ggplot2::aes_string(x = "mean", y = "std")) +
      ggplot2::labs(x = "mean(Sens)", y = "std(Sens)") +
      ggplot2::ggtitle(title)

    if (!is.null(sens_orig$cv)) {
      bootstrapped_mean <- t(apply(sens_orig$boot[orig_order,1,],1, stats::quantile, c(0.05,0.95)))
      bootstrapped_mean <- data.frame('mean_ci_lower' = bootstrapped_mean[,1],
                                      'mean_ci_upper' = bootstrapped_mean[,2],
                                      'std' = sens$std,
                                      'mean' = sens$mean
                                      )
      significance_std <- data.frame('std_min' = sens$std - sens_orig$cv[[1]]$cv,
                                     'std' = sens$std,
                                     'mean' = sens$mean
                                     )

      signif <- sens$meanSensSQ - sens_orig$cv[[2]]$cv[orig_order]
      bootstrapped_mean$color <- ifelse(signif < 0, "black",
                                       ifelse(sens$mean > 0, "chartreuse3", "red"))
      significance_std$color <- ifelse(signif < 0, "black",
                                       ifelse(sens$mean > 0, "chartreuse3", "red"))
      plotlist[[1]] <- plotlist[[1]] +
        ggplot2::geom_errorbarh(data=bootstrapped_mean,
                                ggplot2::aes_string(xmin = "mean_ci_lower",
                                                    xmax = "mean_ci_upper",
                                                    y = "std",
                                                    color = "color")) +
        ggplot2::geom_errorbar(data=significance_std,
                               ggplot2::aes_string(ymin = "std_min",
                                                   ymax = "std",
                                                   x = "mean",
                                                   color = "color")) +
        ggplot2::scale_color_identity()

    }

    plotlist[[1]] <- plotlist[[1]] +
      ggrepel::geom_label_repel(ggplot2::aes_string(x = "mean", y = "std", label = "varNames"),
                                max.overlaps = ifelse(nrow(sens)>10, 2 * nrow(sens), nrow(sens)))

    if (is.null(sens_orig$cv)) {
      plotlist[[2]] <- ggplot2::ggplot(sens) +
        ggplot2::geom_col(ggplot2::aes_string(x = "varNames", y = "meanSensSQ", fill = "mean")) +
        ggplot2::scale_fill_gradient2(
          low='red', mid='black',
          high='chartreuse3', midpoint = 0
        )
    } else {
      sq_data <- data.frame(mean = sens$mean,
                         meanSq = sens$meanSensSQ,
                         cv = signif,
                         Index = row.names(sens))

      sq_data$color <- ifelse(signif < 0, "black",
                              ifelse(sq_data$mean > 0, "chartreuse3", "red"))

      plotlist[[2]] <- ggplot2::ggplot(sq_data,
                                       ggplot2::aes_string(x = "Index",
                                                           y = "meanSq")) +
        ggplot2::geom_point(ggplot2::aes_string()) +
        ggplot2::geom_errorbar(ggplot2::aes_string(ymin = "cv",
                                                   ymax = "meanSq",
                                                   color = "color"),
                               width = 0.2) +
        ggplot2::geom_hline(ggplot2::aes(yintercept = 0), color = "black")  +
        ggplot2::theme(legend.position = "none") +
        ggplot2::scale_color_identity()
    }

    plotlist[[2]] <- plotlist[[2]] +
      ggplot2::labs(x = "Input variables", y = "sqrt(mean(S^2))") +
      ggplot2::guides(fill = "none")

    if (dodge_var) {
      plotlist[[2]] <- plotlist[[2]] +
        ggplot2::scale_x_discrete(guide = ggplot2::guide_axis(n.dodge=ifelse(nrow(sens) > 3, 4 + nrow(sens) %% 2, 2 + nrow(sens) %% 2)))
    }

    if (der) {
      # If the raw values of the derivatives has been passed to the function
      # the density plots of each of these derivatives can be extracted and plotted
      der2 <- as.data.frame(raw_sens)
      names(der2) <- row.names(sens)
      # Remove any variable which is all zero -> pruned variable
      der2 <- der2[,!sapply(der2,function(x){all(x ==  0)}), drop=FALSE]
      if (!is.null(inp_var)) {
        inp_var <- match.arg(inp_var, names(der2), several.ok = TRUE)
      } else {
        inp_var <- names(der2)
      }
      dataplot <- reshape2::melt(der2, measure.vars = inp_var)

      plotlist[[3]] <- ggplot2::ggplot(dataplot) +
        ggplot2::geom_density(ggplot2::aes_string(x = "value", fill = "variable", color = "variable"),
                              alpha = 0.4,
                              bw = "bcv") +
        ggplot2::labs(x = "Sens", y = "density(Sens)")

      # Check the right x limits for the density plots
      quant <- stats::quantile(abs(dataplot$value), c(0.8, 1))
      obtain_quant <- function(serie, quant1, quant2, iter = 0) {
        quants <- stats::quantile(serie, c(quant1, quant2))
        iter <- iter + 1
        if (quants[1] != quants[2] || iter > 500) {
          return(quants)
        } else {
          return(obtain_quant(serie, quant1*0.85, quant2/0.85, iter))
        }
      }
      if (10*quant[1] < quant[2]) { # Distribution has too much dispersion
        xlim <- c(-1,1)*max(abs(obtain_quant(dataplot$value, 0.2, 0.8)))
        if (abs(xlim[1]) < 1e-150) {
          xlim <- c(-1e-150,1e-150)
        }
      } else {
        xlim <- c(-1.1, 1.1)*max(abs(dataplot$value), na.rm = TRUE)
      }
      if (xlim[1] != xlim[2]) {
        plotlist[[3]] <- plotlist[[3]] + ggplot2::xlim(xlim)
      }

      # ggplot2::xlim(-2 * max(sens$std, na.rm = TRUE), 2 * max(sens$std, na.rm = TRUE))
      # Check if ggforce package is installed in the device
      # if it's installed and there are any density distribution that is
      # too small compared with others, make a facet_zoom to show better all distributions
      if (zoom) {
        if (requireNamespace("ggforce")) {
          maxd <- c()
          for (i in 1:ncol(der2)) {
            maxd <- c(maxd, max(stats::density(der2[,i])$y))
          }
          if (max(maxd) > 10*min(maxd)){
            plotlist[[3]] <- plotlist[[3]] + ggforce::facet_zoom(zoom.size = 1, ylim = c(0,1.25*min(maxd)))
          }
        }
      }
      plotlist[[3]] <- plotlist[[3]] + ggplot2::theme(legend.position='bottom')
      if (quit.legend) {
        plotlist[[3]] <- plotlist[[3]] +
          ggplot2::theme(legend.position = "none")
      }
    }
    pl[[out]] <- plotlist
  }
  if (!is.null(plot_type)) {
    plot_type <- match.arg(plot_type, c('mean_sd', 'square', 'raw'), several.ok = TRUE)
    plot_number <- c()
    if ('mean_sd' %in% plot_type) {
      plot_number <- c(plot_number, 1)
    }
    if ('square' %in% plot_type) {
      plot_number <- c(plot_number, 2)
    }
    if ('raw' %in% plot_type) {
      plot_number <- c(plot_number, 3)
    }
  } else {
    plot_number <- c(1, 2, 3)
  }
  # Plot the list of plots created before
  gridExtra::grid.arrange(grobs = pl[[ifelse(is.character(output), which(output == names(sens_orig$sens)), output)]][plot_number],
                          nrow  = length(plot_number),
                          ncols = 1)
  # Return the plots created if the user want to edit them by hand
  return(invisible(plotlist))
}


