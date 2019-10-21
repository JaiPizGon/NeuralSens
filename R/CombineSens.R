#' Sensitivity analysis plot over time of the data
#'
#' @description Plot of sensitivity of the neural network output respect
#' to the inputs over the time variable from the data provided
#' @param object \code{list} of \code{data.frames} with the sensitivity measures or
#' \code{array} with the raw sensitivities calculated with \code{\link[NeuralSens]{SensAnalysisMLP}}
#' @param comb_type if \code{object} is \code{array}, function to combine the third dimension of the array.
#' It can be "mean" or "sqmean". It can also be a function to combine the rows of the array.
#' @return sensitivities of the same type as \code{object} with the combine sensitivities of all outputs
#' @examples
#' \dontrun{
#' # mod should be a neural network classification model
#' sens <- SensAnalysisMLP(mod)
#' combinesens <- CombineSens(sens)
#' rawsens <- SensAnalysisMLP(mod, .rawSens = TRUE)
#' meanCombinerawSens <- CombineSens(rawsens, "mean")
#' sqmeanCombinerawSens <- CombineSens(rawsens, "sqmean")
#' }
#' @export CombineSens
CombineSens <- function(object, comb_type = "mean") {
  if (is.list(object)) {
    # Measures of sensitivity, not raw
    sens_to_return <- object[[1]]
    means <- NULL
    stds <- NULL
    sqmeans <- NULL
    for (i in 1:length(object)) {
      means[[i]] <- object[[i]]$mean
      stds[[i]] <- object[[i]]$std
      sqmeans[[i]] <- object[[i]]$meanSensSQ
    }
    means <- as.data.frame(means)
    stds <- as.data.frame(stds)
    sqmeans <- as.data.frame(sqmeans)
    sens_to_return$mean <- rowMeans(means, na.rm = TRUE)
    # sens_to_return$std <- sqrt(rowMeans(apply(stds, 2, function(x){x^2})))
    # Formula to combine the stds extracted from
    # https://www.researchgate.net/post/How_to_combine_standard_deviations_for_three_groups
    # and assuming that n >> 1 (reasonable if we have trained a neural network)
    sens_to_return$std <- sqrt(rowMeans(apply(stds, 2, function(x){x^2})) +
                                 rowMeans(apply(means, 2, function(x){(x - sens_to_return$mean)^2})))
    sens_to_return$meanSensSQ <- rowMeans(sqrt(sqmeans), na.rm = TRUE) ^ 2
    return(sens_to_return)
  } else if (is.array(object)) {
    # Only accept mean and mean square
    if(is.function(comb_type)) {
      sens_to_return <- apply(object, c(1,2), comb_type)
    } else if (comb_type == "mean") {
      sens_to_return <- apply(object, c(1,2), mean, na.rm = TRUE)
    } else if (comb_type == "sqmean") {
      sens_to_return <- apply(object, c(1,2), function(x){mean(x^2, na.rm = TRUE)})
    } else {
      stop("comb_type must be a function to combine rows")
    }
    sens_to_return <- array(sens_to_return, dim = c(dim(sens_to_return),1))
    colnames(sens_to_return) <- colnames(object)
    return(sens_to_return)
  } else {
    return(object)
  }
}
