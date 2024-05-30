#' Sensitivity alpha-curve associated to MLP function
#'
#' @description Obtain sensitivity alpha-curves associated to MLP function obtained from
#' the sensitivities returned by \code{\link[NeuralSens]{SensAnalysisMLP}}.
#' @param sens sensitivity object returned by \code{\link[NeuralSens]{SensAnalysisMLP}}
#' @param tol difference between M_alpha and maximum sensitivity of the sensitivity of each input variable
#' @param max_alpha maximum alpha value to analyze
#' @param curve_equal_origin make all the curves begin at (1,0)
#' @param inp_var \code{character} indicating which input variable to show in density plot. Only useful when
#' choosing plot_type='raw' to show the density plot of one input variable. If \code{NULL}, all variables
#' are plotted in density plot. By default is \code{NULL}.
#' @param line_width \code{int} width of the line in the plot.
#' @param title \code{char} title of the alpha-curves plot
#' @param alpha_bar \code{int} alpha value to show as column plot.
#' @param kind \code{char} select the type of plot: "line" or "bar"
#' @return alpha-curves of the MLP function
#' @examples
#' \donttest{
#' mod <- RSNNS::mlp(simdata[, c("X1", "X2", "X3")], simdata[, "Y"],
#'                  maxit = 1000, size = 15, linOut = TRUE)
#'
#' sens <- SensAnalysisMLP(mod, trData = simdata,
#'                         output_name = "Y", plot = FALSE)
#'
#' AlphaSensAnalysis(sens)
#' }
#' @export AlphaSensAnalysis
AlphaSensAnalysis <- function(sens,
                              tol = NULL,
                              max_alpha = 15,
                              curve_equal_origin = FALSE,
                              inp_var = NULL,
                              line_width = 1,
                              title='Alpha curve of Lp norm values',
                              alpha_bar = 1,
                              kind = "line"
                              ) {
  if (length(sens$raw_sens) != 1) {
    stop("This analysis is thought for MLPs focused on Regression, it does not work for Classiffication MLPs")
  }

  raw_sens <- sens$raw_sens[[1]]

  if (!is.null(inp_var)) {
    inp_var <- match.arg(inp_var, colnames(raw_sens), several.ok = TRUE)
    raw_sens <- raw_sens[, inp_var, drop=FALSE]
  }
  alpha_curves <- list()
  for (input in 1:ncol(raw_sens)) {
    alpha_curve <- AlphaSensCurve(raw_sens[,input], tol, max_alpha)

    max_sens <- max(abs(raw_sens[,input]))
    if (curve_equal_origin) {
      max_sens <- max_sens - alpha_curve[1]
      alpha_curve <- alpha_curve - alpha_curve[1]
    }
    alpha_curves[[input]] <- data.frame(
      input_var   = colnames(raw_sens)[input],
      alpha_curve = alpha_curve,
      alpha       = 1:length(alpha_curve)
      )

    alpha_curves[[input]] <- rbind(alpha_curves[[input]], data.frame(
      input_var   = colnames(raw_sens)[input],
      alpha_curve = max_sens,
      alpha       = as.numeric('Inf')
    ))
  }
  alpha_curves <- do.call("rbind",alpha_curves)
  if (kind == 'line'){
    # Let's create a new data frame for the max sensitivity points
    max_sens_df <- alpha_curves %>%
      dplyr::filter(alpha_curves$alpha == Inf) %>%
      dplyr::mutate(
        xbeg = max_alpha,
        xend = max_alpha * 1.1,
        xend2 = max_alpha * 1.12
        )

    for (input in unique(alpha_curves$input_var)) {
      max_sens_df[max_sens_df$input_var == input, 'alpha_prev'] = max(alpha_curves[!is.infinite(alpha_curves$alpha) & (alpha_curves$input_var == input) ,'alpha_curve'])
    }


    alpha_curves <- alpha_curves %>%
      dplyr::filter(!(alpha_curves$alpha == Inf))

    breaks <- c(seq_len(max_alpha), max_alpha * 1.12)
    labels <- c(as.character(seq_len(max_alpha)), expression(infinity))
    # Create the plot for alpha curves with segments extending to the right
    p <- ggplot2::ggplot(alpha_curves,
                         ggplot2::aes_string(x="alpha", y="alpha_curve", color="input_var", group="input_var")) +
      ggplot2::geom_line(linewidth=1) +
      ggplot2::geom_segment(data = max_sens_df,
                            ggplot2::aes_string(x="xbeg", xend="xend",
                                                y="alpha_prev", yend="alpha_curve",
                                                color="input_var"),
                            linetype="dotted", linewidth=1) +
      ggplot2::geom_segment(data=max_sens_df,
                            ggplot2::aes_string(x="xend", xend="xend2",
                                                y="alpha_curve", yend="alpha_curve",
                                                color="input_var"), linewidth=1) +
      ggplot2::labs(x = expression(alpha), y = expression("(m"*s[italic("X")*","~italic("j")]^alpha*"(f))"),
                    color='Input') +
      ggrepel::geom_text_repel(
        data = max_sens_df,
        ggplot2::aes_string(label="input_var", x="xend2", y="alpha_curve"),
        nudge_x = 0.5,
        direction = 'y'
      ) +
      ggbreak::scale_x_break(c(max_alpha * 1.01, max_alpha * 1.09), space = 0.3) +
      ggplot2::scale_x_continuous(breaks = breaks, labels = labels) +
      ggplot2::ggtitle(title)

    print(p)
    return(invisible(p))
  } else if(kind == 'bar') {
    if (alpha_bar == 'inf') {
      alpha_df <- alpha_curves[alpha_curves$alpha == max(alpha_curves$alpha),]
    } else {
      alpha_df <- alpha_curves[alpha_curves$alpha == alpha_bar,]
    }
    p <- ggplot2::ggplot(alpha_df) +
      ggplot2::geom_col(ggplot2::aes_string(x = 'input_var', y = 'alpha_curve', fill = 'input_var')) +
      ggplot2::labs(x = "Input variables", y = expression("(m"*s[italic("X")*","~italic("j")]^alpha*"(f))")) +
      ggplot2::ggtitle(title) +
      ggplot2::theme(legend.position = "none")

    print(p)
    return(invisible(p))
  }
}

#' Sensitivity alpha-curve associated to MLP function of an input variable
#'
#' @description Obtain sensitivity alpha-curve associated to MLP function obtained from
#' the sensitivities returned by \code{\link[NeuralSens]{SensAnalysisMLP}} of an input variable.
#' @param sens raw sensitivities of the MLP output with respect to input variable.
#' @param tol difference between M_alpha and maximum sensitivity of the sensitivity of each input variable
#' @param max_alpha maximum alpha value to analyze
#' @return alpha-curve of the MLP function
#' @examples
#' \donttest{
#' mod <- RSNNS::mlp(simdata[, c("X1", "X2", "X3")], simdata[, "Y"],
#'                  maxit = 1000, size = 15, linOut = TRUE)
#'
#' sens <- SensAnalysisMLP(mod, trData = simdata,
#'                         output_name = "Y", plot = FALSE)
#'
#' AlphaSensCurve(sens$raw_sens[[1]][,1])
#' }
#' @export AlphaSensCurve
AlphaSensCurve <- function(sens, tol = NULL, max_alpha = 100) {
  alpha_curve <- c()
  max_sens <- max(abs(sens))
  tol <- ifelse(is.null(tol), 0.0001 * max_sens, tol)
  alpha <- 0
  N <- length(sens)
  order <- 10^(max(floor(log10(abs(sens))))+1)
  while(alpha < max_alpha) {
    alpha <- alpha + 1
    # Scale alpha by order of magnitude of sens in order to avoid infinite values
    alpha_curve <- c(alpha_curve, order * (sum((abs(sens)/order)^alpha/N))^(1/alpha))
    if (alpha >= 2) {
      if (((max_sens - alpha_curve[alpha]) < tol) || is.infinite(alpha_curve[alpha]) || (alpha_curve[alpha] < alpha_curve[alpha-1])) {
        break
        }
    }
  }

  if (is.infinite(alpha_curve[alpha]) ||  alpha_curve[alpha] == 0) {
    alpha_curve <- alpha_curve[1:(alpha-1)]
  }

  return(alpha_curve)
}
