#' Data frame with 4 variables
#'
#' @description Training dataset with values of temperature and working day to predict electrical demand
#' @name DAILY_DEMAND_TR
#' @author Jose Portela Gonzalez
#' @references
#' Pizarroso J, Portela J, Muñoz A (2022). NeuralSens: Sensitivity Analysis of
#' Neural Networks. Journal of Statistical Software, 102(7), 1-36.
#' @keywords data
#' @format A data frame with 1980 rows and 4 variables:
#' \describe{
#'   \item{DATE}{date of the measure}
#'   \item{DEM}{electrical demand}
#'   \item{WD}{Working Day: index which express how much work is made that day}
#'   \item{TEMP}{weather temperature}
#' }
NULL
#' Data frame with 3 variables
#'
#' @description Validation dataset with values of temperature and working day to predict electrical demand
#' @name DAILY_DEMAND_TV
#' @author Jose Portela Gonzalez
#' @references
#' Pizarroso J, Portela J, Muñoz A (2022). NeuralSens: Sensitivity Analysis of
#' Neural Networks. Journal of Statistical Software, 102(7), 1-36.
#' @keywords data
#' @format A data frame with 7 rows and 3 variables:
#' \describe{
#'   \item{DATE}{date of the measure}
#'   \item{WD}{Working Day: index which express how much work is made that day}
#'   \item{TEMP}{weather temperature}
#' }
NULL
#' Simulated data to test the package functionalities
#'
#' @description \code{data.frame} with 2000 rows of 4 columns with 3
#' input variables \code{X1, X2, X3} and one output variable \code{Y}.
#' The data is already scaled, and has been generated using the following code:
#'
#' \code{set.seed(150)}
#'
#'
#' \code{simdata <- data.frame(}
#' \code{   "X1" = rnorm(2000, 0, 1),}
#' \code{   "X2" = rnorm(2000, 0, 1),}
#' \code{   "X3" = rnorm(2000, 0, 1)}
#' \code{ )}
#'
#'
#' \code{ simdata$Y <- simdata$X1^2 + 0.5*simdata$X2 + 0.1*rnorm(2000, 0, 1)}
#'
#'
#' @name simdata
#' @author Jaime Pizarroso Gonzalo
#' @references
#' Pizarroso J, Portela J, Muñoz A (2022). NeuralSens: Sensitivity Analysis of
#' Neural Networks. Journal of Statistical Software, 102(7), 1-36.
#' @keywords data
#' @format A data frame with 2000 rows and 4 variables:
#' \describe{
#'   \item{X1}{Random input 1}
#'   \item{X2}{Random input 2}
#'   \item{X3}{Random input 3}
#'   \item{Y}{Output}
#' }
NULL
