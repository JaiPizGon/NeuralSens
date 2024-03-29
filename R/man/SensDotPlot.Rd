% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/SensDotPlot.R
\name{SensDotPlot}
\alias{SensDotPlot}
\title{Sensitivity scatter plot against input values}
\usage{
SensDotPlot(
  object,
  fdata = NULL,
  input_vars = "all",
  output_vars = "all",
  smooth = FALSE,
  nspline = NULL,
  color = NULL,
  grid = FALSE,
  ...
)
}
\arguments{
\item{object}{fitted neural network model or \code{array} containing the raw
sensitivities from the function \code{\link[NeuralSens]{SensAnalysisMLP}}}

\item{fdata}{\code{data.frame} containing the data to evaluate the sensitivity of the model.}

\item{input_vars}{\code{character vector} with the variables to create the scatter plot. If \code{"all"},
then scatter plots are created for all the input variables in \code{fdata}.}

\item{output_vars}{\code{character vector} with the variables to create the scatter plot. If \code{"all"},
then scatter plots are created for all the output variables in \code{fdata}.}

\item{smooth}{\code{logical} if \code{TRUE}, \code{geom_smooth} plots are added to each variable plot}

\item{nspline}{\code{integer} if \code{smooth} is TRUE, this determine the degree of the spline used
to perform \code{geom_smooth}. If \code{nspline} is NULL, the square root of the length of the data
is used as degrees of the spline.}

\item{color}{\code{character} specifying the name of a \code{numeric} variable of \code{fdata} to color the scatter plot.}

\item{grid}{\code{logical}. If \code{TRUE}, plots created are show together using \code{\link[gridExtra]{arrangeGrob}}}

\item{...}{further arguments that should be passed to  \code{\link[NeuralSens]{SensAnalysisMLP}} function}
}
\value{
list of \code{geom_point} plots for the inputs variables representing the
sensitivity of each output respect to the inputs
}
\description{
Plot of sensitivities of the neural network output respect
to the inputs
}
\examples{
## Load data -------------------------------------------------------------------
data("DAILY_DEMAND_TR")
fdata <- DAILY_DEMAND_TR
## Parameters of the NNET ------------------------------------------------------
hidden_neurons <- 5
iters <- 250
decay <- 0.1

################################################################################
#########################  REGRESSION NNET #####################################
################################################################################
## Regression dataframe --------------------------------------------------------
# Scale the data
fdata.Reg.tr <- fdata[,2:ncol(fdata)]
fdata.Reg.tr[,3] <- fdata.Reg.tr[,3]/10
fdata.Reg.tr[,1] <- fdata.Reg.tr[,1]/1000

# Normalize the data for some models
preProc <- caret::preProcess(fdata.Reg.tr, method = c("center","scale"))
nntrData <- predict(preProc, fdata.Reg.tr)

#' ## TRAIN nnet NNET --------------------------------------------------------
# Create a formula to train NNET
form <- paste(names(fdata.Reg.tr)[2:ncol(fdata.Reg.tr)], collapse = " + ")
form <- formula(paste(names(fdata.Reg.tr)[1], form, sep = " ~ "))

set.seed(150)
nnetmod <- nnet::nnet(form,
                      data = nntrData,
                      linear.output = TRUE,
                      size = hidden_neurons,
                      decay = decay,
                      maxit = iters)
# Try SensDotPlot
NeuralSens::SensDotPlot(nnetmod, fdata = nntrData)
}
