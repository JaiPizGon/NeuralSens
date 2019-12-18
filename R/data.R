#' Data frame with 4 variables
#'
#' @description Training dataset with values of temperature and working day to predict electrical demand
#' @name DAILY_DEMAND_TR
#' @doctype data
#' @author Jose Portela Gonzalez
#' @keywords data
#' @format A data frame with 1980 rows and 4 variables:
#' \describe{
#'   \item{fecha}{date of the measure}
#'   \item{DEM}{electrical demand}
#'   \item{WD}{Working Day: index which express how much work is made that day}
#'   \item{TEMP}{weather temperature}
#' }
NULL
#' Data frame with 3 variables
#'
#' @description Validation dataset with values of temperature and working day to predict electrical demand
#' @name DAILY_DEMAND_TV
#' @doctype data
#' @author Jose Portela Gonzalez
#' @keywords data
#' @format A data frame with 7 rows and 3 variables:
#' \describe{
#'   \item{fecha}{date of the measure}
#'   \item{WD}{Working Day: index which express how much work is made that day}
#'   \item{TEMP}{weather temperature}
#' }
NULL
#' List of 4 dataframes to test the functions with different variables types
#'
#' @description List of 4 dataframes to test the functions with different
#' variables types (numeric and character output and inputs)
#' @name syntheticdata
#' @doctype data
#' @author Jose Portela Gonzalez
#' @keywords data
#' @format list of 4 data.frames with 4 columns for 3 inputs and one output:
#' \describe{
#'   \item{RegOutNumInp}{data.frame}
#'   \itemize{
#'       \item{X1} {Input 1 of the subset 1 (numeric)}
#'       \item{X2} {Input 2 of the subset 1 (numeric)}
#'       \item{X3} {Input 3 of the subset 1 (numeric)}
#'       \item{Y} {Output of the subset 1 (numeric)}
#'   }
#'   \item{ClsOutNumInp}{data.frame}
#'   \itemize{
#'       \item{X1} {Input 1 of the subset 2 (numeric)}
#'       \item{X2} {Input 2 of the subset 2 (numeric)}
#'       \item{X3} {Input 3 of the subset 2 (numeric)}
#'       \item{Y} {Output of the subset 2 (character)}
#'   }
#'   \item{ClsOutClsInp}{data.frame}
#'   \itemize{
#'       \item{X1} {Input 1 of the subset 3 (character)}
#'       \item{X2} {Input 2 of the subset 3 (numeric)}
#'       \item{X3} {Input 3 of the subset 3 (numeric)}
#'       \item{Y} {Output of the subset 3 (character)}
#'   }
#'   \item{ClsOutClsInp}{data.frame}
#'   \itemize{
#'       \item{X1} {Input 1 of the subset 4 (numeric)}
#'       \item{X2} {Input 2 of the subset 4 (character)}
#'       \item{X3} {Input 3 of the subset 4 (numeric)}
#'       \item{Y} {Output of the subset 4 (numeric)}
#'   }
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
#' @doctype data
#' @author Jaime Pizarroso Gonzalo
#' @keywords data
#' @format A data frame with 2000 rows and 4 variables:
#' \describe{
#'   \item{X1}{Random input 1}
#'   \item{X2}{Random input 2}
#'   \item{X3}{Random input 3}
#'   \item{Y}{Output}
#' }
NULL
