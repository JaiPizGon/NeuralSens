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
#'   \item{WD}{working day coefficient}
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
#'   \item{WD}{working day coefficient}
#'   \item{TEMP}{weather temperature}
#' }
NULL
#' Data frame with 15 variables
#'
#' @description Dataset composed of several inputs with its corresponding outputs to test the
#' functionality of the package for regression
#' @name syntheticdata
#' @doctype data
#' @author Jose Portela Gonzalez
#' @keywords data
#' @format A data frame with 1000 rows and 15 variables:
#' \describe{
#'   \item{P1x1}{Input 1 of the subset 1 (numeric)}
#'   \item{P1x2}{Input 2 of the subset 1 (numeric)}
#'   \item{P1x3}{Input 3 of the subset 1 (numeric)}
#'   \item{P1y}{Output of the subset 1 (numeric)}
#'   \item{P2x1}{Input 1 of the subset 2 (numeric)}
#'   \item{P2x2}{Input 2 of the subset 2 (numeric)}
#'   \item{P2x3}{Input 3 of the subset 2 (numeric)}
#'   \item{P2y}{Output of the subset 2 (character)}
#'   \item{P3x1}{Input 1 of the subset 3 (numeric)}
#'   \item{P3x2}{Input 2 of the subset 3 (numeric)}
#'   \item{P3x3}{Input 3 of the subset 3 (numeric)}
#'   \item{P3y}{Output of the subset 3 (numeric)}
#'   \item{P4x1}{Input 1 of the subset 3 (numeric)}
#'   \item{P4x2}{Input 2 of the subset 3 (character)}
#'   \item{P4y}{Output of the subset 3 (numeric)}
#' }
NULL
