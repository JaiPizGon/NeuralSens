#' Sensitivity alpha-curve associated to MLP function
#'
#' @description Obtain sensitivity alpha-curves associated to MLP function obtained from
#' the sensitivities returned by \code{\link[NeuralSens]{SensAnalysisMLP}}.
#' @param sens sensitivity object returned by \code{\link[NeuralSens]{SensAnalysisMLP}}
#' @param tol difference between M_alpha and maximum sensitivity of the sensitivity of each input variable
#' @param max_alpha maximum alpha value to analyze
#' @param interpolate_alpha interpolate alpha mean if difference of maximum sensitivity
#'  and last alpha evaluated is less than \code{tol}
#' @param curve_equal_length make all the curves of the same length
#' @param curve_equal_origin make all the curves begin at (1,0)
#' @param curve_divided_max create second plot of curves divided by maximum alpha
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
AlphaSensAnalysis <- function(sens, tol = NULL, max_alpha = 100, interpolate_alpha = FALSE, curve_equal_length = FALSE, curve_equal_origin = FALSE, curve_divided_max = FALSE) {
  if (length(sens$raw_sens) != 1) {
    stop("This analysis is thought for MLPs focused on Regression, it does not work for Classiffication MLPs")
  }
  raw_sens <- sens$raw_sens[[1]]
  alpha_curves <- list()
  max_alpha_len <- 0
  for (input in 1:ncol(raw_sens)) {
    alpha_curve <- AlphaSensCurve(raw_sens[,input], tol, max_alpha)

    max_sens <- max(abs(raw_sens[,input]))
    alpha_begin_interpolate = NaN
    if (interpolate_alpha) {
      # Interpolate missing point to reach maximum sensitivity
      alpha_begin_interpolate = alpha_curve[length(alpha_curve)]
      m <- (alpha_curve[length(alpha_curve)] - alpha_curve[length(alpha_curve)-1])
      alpha_curve <- c(alpha_curve, seq(alpha_curve[length(alpha_curve)], max(abs(raw_sens[,input])), m))
    }
    if (curve_equal_origin) {
      max_sens <- max_sens - alpha_curve[1]
      alpha_curve <- alpha_curve - alpha_curve[1]
    }
    alpha_curves[[input]] <- data.frame(
      input_var   = colnames(raw_sens)[input],
      alpha_curve = alpha_curve,
      alpha       = 1:length(alpha_curve),
      alpha_max   = max_sens,
      alpha_bi    = alpha_begin_interpolate
      )

    max_alpha_len <- max(max_alpha_len, length(alpha_curve))
  }
  if (curve_equal_length) {
    # Fill missing alpha in shorter curves
    for (input in 1:ncol(raw_sens)) {
      length_curve <- nrow(alpha_curves[[input]])
      if (length_curve < max_alpha_len) {
        alpha_curves[[input]] <- rbind(
          alpha_curves[[input]],
          data.frame(
            input_var = colnames(raw_sens)[input],
            alpha_curve = alpha_curves[[input]][length_curve,"alpha_curve"],
            alpha = (length_curve+1):max_alpha_len,
            alpha_max = alpha_curves[[input]][length_curve,"alpha_max"],
            alpha_bi = alpha_curves[[input]][length_curve,"alpha_bi"]
          )
        )
      }
    }
  }

  alpha_curves <- do.call("rbind",alpha_curves)
  p1 <- ggplot2::ggplot(alpha_curves) +
    ggplot2::geom_line(ggplot2::aes_string(x = "alpha", y = "alpha_curve", color = "input_var")) +
    ggplot2::geom_hline(ggplot2::aes_string(yintercept = "alpha_bi", color = "input_var"),
                        linetype = "dotted") +
    ggplot2::geom_hline(ggplot2::aes_string(yintercept = "alpha_max", color = "input_var"),
                        linetype = "dashed") +
    ggplot2::ylab("alpha value") +
    ggplot2::ggtitle("Alpha curve of Lp norm values")
  if (curve_divided_max) {
    alpha_curves$divided <- alpha_curves$alpha_curve / alpha_curves$alpha_max
    p2 <- ggplot2::ggplot(alpha_curves) +
      ggplot2::geom_line(ggplot2::aes_string(x = "alpha", y = "divided", color = "input_var")) +
      ggplot2::geom_hline(ggplot2::aes(yintercept = 1),
                          linetype = "dashed") +
      ggplot2::ylab("alpha value")+
      ggplot2::ggtitle("Alpha curve of Lp norm values divided by maximum")
    g <- gridExtra::grid.arrange(grobs=list(p1, p2), ncol = 2)
    plot(g)
    return(invisible(g))
  } else {
    plot(p1)
    return(invisible(p1))
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
