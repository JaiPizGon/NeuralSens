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
