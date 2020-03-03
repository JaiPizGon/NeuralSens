#' Sensitivity of MLP models
#'
#' @description Function for evaluating the sensitivities of the inputs
#'   variables in a mlp model
#' @param MLP.fit fitted neural network model
#' @param trData \code{data.frame} containing the data to evaluate the sensitivity of the model
#' @param actfunc \code{character} vector indicating the activation function of each
#'   neurons layer.
#' @param deractfunc \code{character} vector indicating the derivative of the activation
#' function of each neurons layer.
#' @param .returnSens DEPRECATED
#' @param preProc preProcess structure applied to the training data. See also
#'   \code{\link[caret]{preProcess}}
#' @param terms function applied to the training data to create factors. See
#'   also \code{\link[caret]{train}}
#' @param plot \code{logical} whether or not to plot the analysis. By default is
#'   \code{TRUE}.
#' @param .rawSens DEPRECATED
#' @param output_name \code{character} name of the output variable in order to
#'   avoid changing the name of the output variable in \code{trData} to
#'   '.outcome'
#' @param sens_origin_layer \code{numeric} specifies the layer of neurons with
#'   respect to which the derivative must be calculated. The input layer is
#'   specified by 1 (default).
#' @param sens_end_layer \code{numeric} specifies the layer of neurons of which
#'   the derivative is calculated. It may also be 'last' to specify the output
#'   layer (default).
#' @param sens_origin_input \code{logical} specifies if the derivative must be
#'   calculated with respect to the inputs (\code{TRUE}) or output
#'   (\code{FALSE}) of the \code{sens_origin_layer} layer of the model. By
#'   default is \code{TRUE}.
#' @param sens_end_input \code{logical} specifies if the derivative calculated
#'   is of the output (\code{FALSE}) or from the input (\code{TRUE}) of the
#'   \code{sens_end_layer} layer of the model. By default is \code{FALSE}.
#' @param ...	additional arguments passed to or from other methods
#' @return dataframe with the sensitivities obtained for each variable if
#'   \code{.returnSens = TRUE}. If \code{.returnSens = FALSE}, the sensitivities
#'   without processing are returned in a 3D array. If there is more than one
#'   output, the sensitivities of each output are given in a list.
#' @section Plots: \itemize{ \item Plot 1: colorful plot with the classification
#'   of the classes in a 2D map \item Plot 2: b/w plot with probability of the
#'   chosen class in a 2D map \item Plot 3: plot with the stats::predictions of
#'   the data provided }
#' @details In case of using an input of class \code{factor} and a package which
#'   need to enter the input data as matrix, the dummies must be created before
#'   training the neural network.
#'
#'   After that, the training data must be given to the function using the
#'   \code{trData} argument.
#' @references
#'   \url{https://www.researchgate.net/publication/220577792_Use_of_some_sensitivity_criteria_for_choosing_networks_with_good_generalization_ability}
#'
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
#' # Try SensAnalysisMLP
#' NeuralSens::SensAnalysisMLP(nnetmod, trData = nntrData)
#' \donttest{
#' # Try SensAnalysisMLP to calculate sensitivities with respect to output of hidden neurones
#' NeuralSens::SensAnalysisMLP(nnetmod, trData = nntrData,
#'                              sens_origin_layer = 2,
#'                              sens_end_layer = "last",
#'                              sens_origin_input = FALSE,
#'                              sens_end_input = FALSE)
#' ## Train caret NNET ------------------------------------------------------------
#' # Create trainControl
#' ctrl_tune <- caret::trainControl(method = "boot",
#'                                  savePredictions = FALSE,
#'                                  summaryFunction = caret::defaultSummary)
#' set.seed(150) #For replication
#' caretmod <- caret::train(form = DEM~.,
#'                               data = fdata.Reg.tr,
#'                               method = "nnet",
#'                               linout = TRUE,
#'                               tuneGrid = data.frame(size = 3,
#'                                                     decay = decay),
#'                               maxit = iters,
#'                               preProcess = c("center","scale"),
#'                               trControl = ctrl_tune,
#'                               metric = "RMSE")
#'
#' # Try SensAnalysisMLP
#' NeuralSens::SensAnalysisMLP(caretmod)
#'
#' ## Train h2o NNET --------------------------------------------------------------
#' # Create a cluster with 4 available cores
#' h2o::h2o.init(ip = "localhost",
#'               nthreads = 4,
#'               max_mem_size = "2g")
#'
#' # Reset the cluster
#' h2o::h2o.removeAll()
#' fdata_h2o <- h2o::as.h2o(x = fdata.Reg.tr, destination_frame = "fdata_h2o")
#'
#' set.seed(150)
#' h2omod <-h2o:: h2o.deeplearning(x = names(fdata.Reg.tr)[2:ncol(fdata.Reg.tr)],
#'                                      y = names(fdata.Reg.tr)[1],
#'                                      distribution = "AUTO",
#'                                      training_frame = fdata_h2o,
#'                                      standardize = TRUE,
#'                                      activation = "Tanh",
#'                                      hidden = c(hidden_neurons),
#'                                      stopping_rounds = 0,
#'                                      epochs = iters,
#'                                      seed = 150,
#'                                      model_id = "nnet_h2o",
#'                                      adaptive_rate = FALSE,
#'                                      rate_decay = decay,
#'                                      export_weights_and_biases = TRUE)
#'
#' # Try SensAnalysisMLP
#' NeuralSens::SensAnalysisMLP(h2omod)
#'
#' # Turn off the cluster
#' h2o::h2o.shutdown(prompt = FALSE)
#' rm(fdata_h2o)
#'
#' ## Train neural NNET -----------------------------------------------------------
#' set.seed(150)
#' neuralmod <- neural::mlptrain(as.matrix(nntrData[,2:ncol(nntrData)]),
#'                                    hidden_neurons,
#'                                    as.matrix(nntrData[1]),
#'                                    it=iters,
#'                                    visual=FALSE)
#'
#' # Try SensAnalysisMLP
#' trData <- nntrData
#' NeuralSens::SensAnalysisMLP(neuralmod, trData = trData, output_name = "DEM")
#'
#' ## Train RSNNS NNET ------------------------------------------------------------
#' # Normalize data using RSNNS algorithms
#' trData <- as.data.frame(RSNNS::normalizeData(fdata.Reg.tr))
#' names(trData) <- names(fdata.Reg.tr)
#' set.seed(150)
#' RSNNSmod <-RSNNS::mlp(x = trData[,2:ncol(trData)],
#'                            y = trData[,1],
#'                            size = hidden_neurons,
#'                            linOut = TRUE,
#'                            learnFuncParams=c(decay),
#'                            maxit=iters)
#'
#' # Try SensAnalysisMLP
#' NeuralSens::SensAnalysisMLP(RSNNSmod, trData = trData, output_name = "DEM")
#'
#' ## TRAIN neuralnet NNET --------------------------------------------------------
#' # Create a formula to train NNET
#' form <- paste(names(fdata.Reg.tr)[2:ncol(fdata.Reg.tr)], collapse = " + ")
#' form <- formula(paste(names(fdata.Reg.tr)[1], form, sep = " ~ "))
#'
#' set.seed(150)
#' nnmod <- neuralnet::neuralnet(form,
#'                                    nntrData,
#'                                    linear.output = TRUE,
#'                                    rep = 1,
#'                                    hidden = hidden_neurons,
#'                                    lifesign = "minimal",
#'                                    threshold = 7,
#'                                    stepmax = iters,
#'                                    learningrate = decay,
#'                                    act.fct = "tanh")
#'
#' # Try SensAnalysisMLP
#' NeuralSens::SensAnalysisMLP(nnmod)
#'
#'
#' ## USE DEFAULT METHOD ----------------------------------------------------------
#' NeuralSens::SensAnalysisMLP(RegNNET$caret$finalModel$wts,
#'                             trData = fdata.Reg.tv,
#'                             mlpstr = RegNNET$caret$finalModel$n,
#'                             coefnames = RegNNET$caret$coefnames,
#'                             actfun = c("linear","sigmoid","linear"),
#'                             output_name = "DEM")
#'
#' ################################################################################
#' #########################  CLASSIFICATION NNET #################################
#' ################################################################################
#' ## Regression dataframe --------------------------------------------------------
#' # Scale the data
#' fdata.Reg.cl <- fdata[,2:ncol(fdata)]
#' fdata.Reg.cl[,2:3] <- fdata.Reg.cl[,2:3]/10
#' fdata.Reg.cl[,1] <- fdata.Reg.cl[,1]/1000
#'
#' # Normalize the data for some models
#' preProc <- caret::preProcess(fdata.Reg.cl, method = c("center","scale"))
#' nntrData <- predict(preProc, fdata.Reg.cl)
#'
#' # Factorize the output
#' fdata.Reg.cl$DEM <- factor(round(fdata.Reg.cl$DEM, digits = 1))
#'
#' # Normalize the data for some models
#' preProc <- caret::preProcess(fdata.Reg.cl, method = c("center","scale"))
#' nntrData <- predict(preProc, fdata.Reg.cl)
#'
#' ## Train caret NNET ------------------------------------------------------------
#' # Create trainControl
#' ctrl_tune <- caret::trainControl(method = "boot",
#'                                  savePredictions = FALSE,
#'                                  summaryFunction = caret::defaultSummary)
#' set.seed(150) #For replication
#' caretmod <- caret::train(form = DEM~.,
#'                                 data = fdata.Reg.cl,
#'                                 method = "nnet",
#'                                 linout = FALSE,
#'                                 tuneGrid = data.frame(size = hidden_neurons,
#'                                                       decay = decay),
#'                                 maxit = iters,
#'                                 preProcess = c("center","scale"),
#'                                 trControl = ctrl_tune,
#'                                 metric = "Accuracy")
#'
#' # Try SensAnalysisMLP
#' NeuralSens::SensAnalysisMLP(caretmod)
#'
#' ## Train h2o NNET --------------------------------------------------------------
#' # Create local cluster with 4 available cores
#' h2o::h2o.init(ip = "localhost",
#'               nthreads = 4,
#'               max_mem_size = "2g")
#'
#' # Reset the cluster
#' h2o::h2o.removeAll()
#' fdata_h2o <- h2o::as.h2o(x = fdata.Reg.cl, destination_frame = "fdata_h2o")
#'
#' set.seed(150)
#' h2omod <- h2o::h2o.deeplearning(x = names(fdata.Reg.cl)[2:ncol(fdata.Reg.cl)],
#'                                        y = names(fdata.Reg.cl)[1],
#'                                        distribution = "AUTO",
#'                                        training_frame = fdata_h2o,
#'                                        standardize = TRUE,
#'                                        activation = "Tanh",
#'                                        hidden = c(hidden_neurons),
#'                                        stopping_rounds = 0,
#'                                        epochs = iters,
#'                                        seed = 150,
#'                                        model_id = "nnet_h2o",
#'                                        adaptive_rate = FALSE,
#'                                        rate_decay = decay,
#'                                        export_weights_and_biases = TRUE)
#'
#' # Try SensAnalysisMLP
#' NeuralSens::SensAnalysisMLP(h2omod)
#'
#' # Apaga el cluster
#' h2o::h2o.shutdown(prompt = FALSE)
#' rm(fdata_h2o)
#'
#' ## Train neural NNET -----------------------------------------------------------
#' # set.seed(150)
#' # neuralmod <-mlptrain(as.matrix(nntrData[,2:ncol(nntrData)]),
#' #                           hidden_neurons,
#' #                           as.matrix(nntrData[1]),
#' #                           it=iters,
#' #                           visual=FALSE)
#' #
#' # # Try SensAnalysisMLP
#' # NeuralSens::SensAnalysisMLP(neuralmod, trData = trData)
#'
#' # ## Train RSNNS NNET ------------------------------------------------------------
#' # # Normalize data using RSNNS algorithms
#' # trData <- as.data.frame(RSNNS::normalizeData(fdata.Reg.cl))
#' # names(trData) <- names(fdata.Reg.tr)
#' # set.seed(150)
#' # RSNNSmod <- RSNNS::mlp(x = trData[,2:ncol(trData)],
#' #                      y = trData[,1],
#' #                      size = hidden_neurons,
#' #                      linOut = FALSE,
#' #                      learnFuncParams=c(decay),
#' #                      maxit=iters)
#' #
#' # # Try SensAnalysisMLP
#' # NeuralSens::SensAnalysisMLP(RSNNSmod, trData = trData, output_name = "DEM")
#'
#' ## TRAIN neuralnet NNET --------------------------------------------------------
#' # Create a formula to train NNET
#' # form <- paste(names(fdata.Reg.tr)[2:ncol(fdata.Reg.tr)], collapse = " + ")
#' # form <- formula(paste(names(fdata.Reg.tr)[1], form, sep = " ~ "))
#' #
#' # set.seed(150)
#' # nnmod <- neuralnet(form,
#' #                    nntrData,
#' #                    linear.output = FALSE,
#' #                    rep = 1,
#' #                    hidden = hidden_neurons,
#' #                    lifesign = "minimal",
#' #                    threshold = 4,
#' #                    stepmax = iters,
#' #                    learningrate = decay,
#' #                    act.fct = "tanh")
#' #
#' # # Try SensAnalysisMLP
#' # NeuralSens::SensAnalysisMLP(nnmod)
#'
#' ## TRAIN nnet NNET --------------------------------------------------------
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
#' # Try SensAnalysisMLP
#' NeuralSens::SensAnalysisMLP(nnetmod, trData = nntrData)
#' }
#' @export
#' @rdname SensAnalysisMLP
SensAnalysisMLP <- function(MLP.fit,
                            .returnSens = TRUE,
                            plot = TRUE,
                            .rawSens = FALSE,
                            sens_origin_layer = 1,
                            sens_end_layer = "last",
                            sens_origin_input = TRUE,
                            sens_end_input = FALSE,
                            ...) UseMethod('SensAnalysisMLP', MLP.fit)

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP default
SensAnalysisMLP.default <- function(MLP.fit,
                                    .returnSens = TRUE,
                                    plot = TRUE,
                                    .rawSens = FALSE,
                                    sens_origin_layer = 1,
                                    sens_end_layer = "last",
                                    sens_origin_input = TRUE,
                                    sens_end_input = FALSE,
                                    trData,
                                    actfunc = NULL,
                                    deractfunc = NULL,
                                    preProc = NULL,
                                    terms = NULL,
                                    output_name = NULL,
                                    ...) {
  ### Things needed for calculating the sensibilities:
  #   - Structure of the model  -> MLP.fit$n
  #   - Weights of the model    -> MLP.fit$wts
  #   - Name of the inputs      -> MLP.fit$coefnames
  #   - trData [output + inputs], output's name must be .outcome

  # Obtain structure of fitted model
  mlpstr <- MLP.fit$n

  # Obtain weights
  nwts <- NeuralNetTools::neuralweights(MLP.fit$wts, struct = mlpstr)
  wts <- nwts$wts

  # VariableNames
  varnames <- MLP.fit$coefnames
  # Correct varnames
  varnames[which(substr(varnames,1,1) == "`")] <- substr(varnames[which(substr(varnames,1,1) == "`")],
                                                         2,nchar(varnames[which(substr(varnames,1,1) == "`")])-1)
  # Check if the output_name has been specified
  if (!".outcome" %in% names(trData)) {
    if (!is.null(output_name)) {
      names(trData)[names(trData) == output_name] <- ".outcome"
    } else {
      stop("Output variable has not been found in trData")
    }
  }
  # TestData
  dummies <-
    caret::dummyVars(
      .outcome ~ .,
      data = trData,
      fullRank = TRUE,
      sep = NULL
    )

  if (!is.null(terms)) {
    dummies$terms <- terms
  }

  TestData <- data.frame(stats::predict(dummies, newdata = trData))
  # Check names of the variables and correct is there any error
  for(i in which(!names(TestData) %in% varnames)) {
    # Check which varname is different
    for (j in which(!varnames %in% names(TestData))){
      # The problem with the names are with the variables that are factors
      # that they store the names with different notations
      # In the following code we check if the variables are equal without
      # the notation quotes
      a <- unlist(strsplit(names(TestData)[i],""))
      b <- unlist(strsplit(varnames[j],""))
      if(all(a[stats::na.omit(pmatch(b,a))] == b[stats::na.omit(pmatch(a,b))])){
        names(TestData)[i] <- varnames[j]
      }
    }
  }

  TestData <- TestData[, varnames]
  if (!is.null(preProc)) {
    TestData <- stats::predict(preProc, TestData[, varnames])
  }

  # Intermediate structures with data necessary to calculate the structures above
  #    - Z stores the values just before entering the neuron, i.e., sum(weights*inputs)
  #    for each layer of neurons
  #    - O stores the output values of each layer of neurons
  #    - D stores the derivative of the output values of each layer of neurons (Jacobian)
  #    - W stores the weights of the inputs of each layer of neurons
  Z <- list()
  O <- list()
  W <- list()
  D <- list()

  # Initialize the activation and the derivative of the activation function for each layer
  if (is.null(deractfunc)) deractfunc <- actfunc
  ActivationFunction <- lapply(actfunc, NeuralSens::ActFunc)
  DerActivationFunction <- lapply(deractfunc, NeuralSens::DerActFunc)

  W[[1]] <- diag(ncol(TestData)+1)
  # For each row in the TestData
  Z[[1]] <- as.matrix(TestData)
  O[[1]] <- ActivationFunction[[1]](Z[[1]])
  D[[1]] <- array(diag(mlpstr[1]),
                  dim=c(mlpstr[1],
                        mlpstr[1],
                        nrow(TestData)))
  for (irow in 1:nrow(TestData)) {
    D[[1]][,,irow] <- DerActivationFunction[[1]](Z[[1]][irow,])
  }
  # For each layer, calculate the input to the activation functions of each layer
  # This inputs are gonna be used to calculate the derivatives and the output of each layer
  for (l in 2:length(mlpstr)){
    W[[l]] <- data.matrix(as.data.frame(wts[(sum(mlpstr[1:(l-1)])-mlpstr[1]+1):(sum(mlpstr[1:l])-mlpstr[1])]))
    Z[[l]] <- cbind(1, O[[l-1]]) %*% W[[l]]
    O[[l]] <- ActivationFunction[[l]](Z[[l]])
    D[[l]] <- array(diag(mlpstr[l]),
                    dim=c(mlpstr[l],
                          mlpstr[l],
                          nrow(TestData)))
    for (irow in 1:nrow(TestData)) {
      D[[l]][,,irow] <- DerActivationFunction[[l]](Z[[l]][irow,])
    }
  }

  args <- list(...)

  if (!"return_all_sens" %in% names(args[[1]])) {
    out <- structure(list(
      sens = NULL,
      raw_sens = NULL,
      layer_derivatives = D,
      mlp_struct = mlpstr,
      mlp_wts = W,
      layer_origin = sens_origin_layer,
      layer_origin_input = sens_origin_input,
      layer_end = sens_end_layer,
      layer_end_input = sens_end_input,
      trData = trData,
      coefnames = varnames,
      output_name = output_name
    ),
    class = "SensMLP")

    out <- ComputeSensMeasures(out)

    if (plot) {
      # show plots if required
      args <- list(...)
      zoom <- TRUE
      quit.legend <- FALSE
      der <- TRUE
      if ("zoom" %in% names(args[[1]])) {
        zoom <- args[[1]]$zoom
      }
      if ("quit.legend" %in% names(args[[1]])) {
        quit.legend <- args[[1]]$quit.legend
      }
      if ("der" %in% names(args[[1]])) {
        der <- args[[1]]$der
      }
      NeuralSens::SensitivityPlots(out, der, zoom, quit.legend)
    }
    return(out)
  } else {

    # Calculate derivatives with respect with the last layer's output
    d <- list()
    k <- 2 * length(D) - 1
    d[[2 * length(D)]] <- array(diag(dim(D[[length(D)]])[1]),
                                dim = dim(D[[length(D)]]))
    for (l in length(D):1) {
      d[[k]] <- array(NA, dim=c(dim(D[[(k+1)/2]])[2], dim(D[[length(D)]])[2], dim(D[[1]])[3]))
      if (k > 1) {
        d[[k - 1]] <- array(NA, dim=c(dim(W[[(k+1)/2]])[1]-1, dim(D[[length(D)]])[2], dim(D[[1]])[3]))
      }
      for (irow in 1:dim(D[[1]])[3]) {
        d[[k]][,,irow] <- D[[l]][,,irow] %*% d[[k + 1]][,,irow]
        if (k > 1) {
          d[[k - 1]][,,irow] <- W[[l]][2:nrow(W[[l]]),, drop = FALSE] %*% d[[k]][,,irow]
        }
      }
      k <- k - 2
    }
    d <- d[as.logical(1:length(d) %% 2)]

    # Return all the derivatives and the weights of the net
    return(list(d, MLP.fit$n, MLP.fit$wts, varnames, output_name))
  }
}

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP train
SensAnalysisMLP.train <- function(MLP.fit,
                                  .returnSens = TRUE,
                                  plot = TRUE,
                                  .rawSens = FALSE,
                                  sens_origin_layer = 1,
                                  sens_end_layer = "last",
                                  sens_origin_input = TRUE,
                                  sens_end_input = FALSE,
                                  ...) {
  args <- list(...)
  SensAnalysisMLP(MLP.fit$finalModel,
                  trData = if ("trData" %in% names(args)) {args$trData} else {MLP.fit$trainingData},
                  .returnSens = .returnSens,
                  .rawSens = .rawSens,
                  sens_origin_layer = sens_origin_layer,
                  sens_end_layer = sens_end_layer,
                  sens_origin_input = sens_origin_input,
                  sens_end_input = sens_end_input,
                  preProc = if ("preProc" %in% names(args)) {args$preProc} else {MLP.fit$preProcess},
                  terms = if ("terms" %in% names(args)) {args$terms} else {MLP.fit$terms},
                  plot = plot,
                  args[!names(args) %in% c("trData","preProc","terms")])
}

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP H2OMultinomialModel
SensAnalysisMLP.H2OMultinomialModel <- function(MLP.fit,
                                                .returnSens = TRUE,
                                                plot = TRUE,
                                                .rawSens = FALSE,
                                                sens_origin_layer = 1,
                                                sens_end_layer = "last",
                                                sens_origin_input = TRUE,
                                                sens_end_input = FALSE,
                                                ...) {
  args <- list(...)
  createDummiesH2O <- function(trData) {
    # H2O dummies create all levels of factor variables and an extra level for missing (NA).
    # This is different in caret where it creates all levels minus 1 of factor variables.
    # To avoid compatibility problem, we create the dummies of H2O models here,
    # so the caret dummies will just copy the dummies variables here created.
    # Deleting the extra dummies that H2O creates is not feasible due to the
    # neural network of h2o having input neurons for that inputs.

    # Obtain which are the factor variables
    nfactor <- as.data.frame(trData[,sapply(trData,is.factor)])
    if (length(nfactor) > 0) {
      names(nfactor) <- names(trData)[sapply(trData,is.factor)]
      # Obtain dummies with all the levels
      dumm <- fastDummies::dummy_columns(trData)
      # Check if variables with NAs have been created
      # Number of columns in dummies must be the same as trData + levels + 1 missing variable per factor variable
      if (ncol(dumm) != ncol(trData) + sum(lengths(lapply(nfactor,levels))) + ncol(nfactor)){
        for (i in 1:ncol(nfactor)) {
          search <- paste0(names(nfactor)[i], ".missing.NA.")
          if (!search %in% names(dumm)) {
            dumm[,eval(search)] <- 0
          }
        }
      }
      for (i in 1:ncol(nfactor)) {
        for (namespos in which(names(nfactor)[i] == substr(names(dumm),1,nchar(names(nfactor)[i])) & names(nfactor)[i] != names(dumm))) {
          prevname <- names(dumm)[namespos]
          stringr::str_sub(prevname,nchar(names(nfactor)[i])+1,nchar(names(nfactor)[i])+1) <- "."
          names(dumm)[namespos] <- prevname
        }
      }

      # Remove factor variables duplicated
      dumm[,names(nfactor)] <- NULL
      return(dumm)
    } else {
      return(trData)
    }
  }
  # Create empty final model
  finalModel <- NULL
  finalModel$n <- MLP.fit@model$model_summary$units
  # Try to establish connection with the h2o cluster
  out <- tryCatch(
    {h2o::h2o.getConnection()},
    error=function(cond) {
      stop("There is not an active h2o cluster, run h2o.init()\n")
    },
    finally={}
  )
  # Try to extract the weights of the neural network
  wts <- c()
  for (i in 1:(length(c(finalModel$n))-1)) {
    wtsi <- as.data.frame(h2o::h2o.weights(MLP.fit,matrix_id=i))
    bi <- as.data.frame(h2o::h2o.biases(MLP.fit,vector_id=i))
    wtsi <- cbind(bi,wtsi)
    for (j in 1:nrow(wtsi)) {
      wts <- unlist(c(wts, wtsi[j,]))
    }
  }
  if (is.null(wts)) {
    stop("No weights have been detected
         Use argument export_weights_and_biases = TRUE when training the nnet")
  }
  finalModel$wts <- wts

  # Try to extract the training data of the model
  trData <- tryCatch({
    if ("trData" %in% names(args)) {
      args$trData
    } else {
      as.data.frame(eval(parse(text = MLP.fit@parameters$training_frame)))
    }
  }, error=function(cond) {
    stop("The training data has not been detected, load the data to the h2o cluster\n")
  },
  finally={}
  )
  # Change the name of the output in trData
  if(MLP.fit@parameters$y %in% names(trData)) {
    names(trData)[which(MLP.fit@parameters$y == names(trData))] <- ".outcome"
  } else if (!".outcome" %in% names(trData)) {
    stop(paste0("Output ",MLP.fit@parameters$y," has not been found in training data"))
  }
  trData <- trData[,c(".outcome",MLP.fit@parameters$x)]
  # Create the preprocess of center and scale that h2o do automatically
  preProc <- caret::preProcess(trData[,MLP.fit@parameters$x], method = c("center","scale"))
  # Create dummies before calling the default
  copy <- trData
  trData <- createDummiesH2O(trData[,2:ncol(trData)])
  # Order data as weights
  trData <- trData[,names(wts)[2:(ncol(trData)+1)]]
  finalModel$coefnames <- names(trData)
  trData <- cbind(.outcome = copy$.outcome,trData)
  # Create vector of activation functions
  PrepActFuncs <- function(acfun) {
    # Switch case to define which value it returns
    switch(acfun,
           Input = {
             return("linear")
           },
           Tanh = {
             return("tanh")
           },
           Linear = {
             return("linear")
           },
           TanhDropout = {
             return("tanh")
           },
           Rectifier = {
             return("ReLU")
           },
           RectifierDropout = {
             return("ReLU")
           },
           Maxout = {
             stop("SensAnalysisMLP function is not ready for Maxout layers")
           },
           MaxoutDropout = {
             stop("SensAnalysisMLP function is not ready for Maxout layers")
           },
           Softmax = {
             return("softmax")
           },
           {
             stop("SensAnalysisMLP is not ready for the activation function used")
           })
  }
  actfun <- sapply(MLP.fit@model$model_summary$type, PrepActFuncs)
  # Call to the default function
  SensAnalysisMLP.default(finalModel,
                          trData = trData,
                          actfunc = actfun,
                          .returnSens = .returnSens,
                          .rawSens = .rawSens,
                          sens_origin_layer = sens_origin_layer,
                          sens_end_layer = sens_end_layer,
                          sens_origin_input = sens_origin_input,
                          sens_end_input = sens_end_input,
                          preProc = preProc,
                          terms = NULL,
                          plot = plot,
                          output_name = if("output_name" %in% names(args)){args$output_name}else{MLP.fit@parameters$y},
                          deractfunc = if("deractfunc" %in% names(args)){args$deractfunc}else{NULL},
                          args[!names(args) %in% c("trData","output_name","deractfunc")])
  }

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP H2ORegressionModel
SensAnalysisMLP.H2ORegressionModel <- function(MLP.fit,
                                               .returnSens = TRUE,
                                               plot = TRUE,
                                               .rawSens = FALSE,
                                               sens_origin_layer = 1,
                                               sens_end_layer = "last",
                                               sens_origin_input = TRUE,
                                               sens_end_input = FALSE,
                                               ...) {
  args <- list(...)
  createDummiesH2O <- function(trData) {
    # H2O dummies create all levels of factor variables and an extra level for missing (NA).
    # This is different in caret where it creates all levels minus 1 of factor variables.
    # To avoid compatibility problem, we create the dummies of H2O models here,
    # so the caret dummies will just copy the dummies variables here created.
    # Deleting the extra dummies that H2O creates is not feasible due to the
    # neural network of h2o having input neurons for that inputs.

    # Obtain which are the factor variables
    nfactor <- as.data.frame(trData[,sapply(trData,is.factor)])
    if (length(nfactor) > 0) {
      names(nfactor) <- names(trData)[sapply(trData,is.factor)]
      # Obtain dummies with all the levels
      dumm <- fastDummies::dummy_columns(trData)
      # Check if variables with NAs have been created
      # Number of columns in dummies must be the same as trData + levels + 1 missing variable per factor variable
      if (ncol(dumm) != ncol(trData) + sum(lengths(lapply(nfactor,levels))) + ncol(nfactor)){
        for (i in 1:ncol(nfactor)) {
          search <- paste0(names(nfactor)[i], ".missing.NA.")
          if (!search %in% names(dumm)) {
            dumm[,eval(search)] <- 0
          }
        }
      }
      for (i in 1:ncol(nfactor)) {
        for (namespos in which(names(nfactor)[i] == substr(names(dumm),1,nchar(names(nfactor)[i])) & names(nfactor)[i] != names(dumm))) {
          prevname <- names(dumm)[namespos]
          stringr::str_sub(prevname,nchar(names(nfactor)[i])+1,nchar(names(nfactor)[i])+1) <- "."
          names(dumm)[namespos] <- prevname
        }
      }

      # Remove factor variables duplicated
      dumm[,names(nfactor)] <- NULL
      return(dumm)
    } else {
      return(trData)
    }
  }
  # Create empty final model
  finalModel <- NULL
  finalModel$n <- MLP.fit@model$model_summary$units
  # Try to establish connection with the h2o cluster
  out <- tryCatch(
    {h2o::h2o.getConnection()},
    error=function(cond) {
      stop("There is not an active h2o cluster, run h2o.init()\n")
    },
    finally={}
  )
  # Try to extract the weights of the neural network
  wts <- c()
  for (i in 1:(length(c(finalModel$n))-1)) {
    wtsi <- as.data.frame(h2o::h2o.weights(MLP.fit,matrix_id=i))
    bi <- as.data.frame(h2o::h2o.biases(MLP.fit,vector_id=i))
    wtsi <- cbind(bi,wtsi)
    for (j in 1:nrow(wtsi)) {
      wts <- unlist(c(wts, wtsi[j,]))
    }
  }
  if (is.null(wts)) {
    stop("No weights have been detected
         Use argument export_weights_and_biases = TRUE when training the nnet")
  }
  finalModel$wts <- wts
  # Try to extract the training data of the model
  trData <- tryCatch({
    if ("trData" %in% names(args)) {
      args$trData
    } else {
      as.data.frame(eval(parse(text = MLP.fit@parameters$training_frame)))
    }
  }, error=function(cond) {
    stop("The training data has not been detected, load the data to the h2o cluster\n")
  },
  finally={}
  )
  # Change the name of the output in trData
  if(MLP.fit@parameters$y %in% names(trData)) {
    names(trData)[which(MLP.fit@parameters$y == names(trData))] <- ".outcome"
  } else if (!".outcome" %in% names(trData)) {
    stop(paste0("Output ",MLP.fit@parameters$y," has not been found in training data"))
  }
  trData <- trData[,c(".outcome",MLP.fit@parameters$x)]
  # Create the preprocess of center and scale that h2o do automatically
  preProc <- caret::preProcess(trData[,MLP.fit@parameters$x], method = c("center","scale"))
  # Create dummies before calling the default
  copy <- trData
  trData <- createDummiesH2O(trData[,2:ncol(trData)])
  # Order data as weights
  trData <- trData[,names(wts)[2:(ncol(trData)+1)]]
  finalModel$coefnames <- names(trData)
  trData <- cbind(.outcome = copy$.outcome,trData)
  # Create vector of activation functions
  PrepActFuncs <- function(acfun) {
    # Switch case to define which value it returns
    switch(acfun,
           Input = {
             return("linear")
           },
           Tanh = {
             return("tanh")
           },
           Linear = {
             return("linear")
           },
           TanhDropout = {
             return("tanh")
           },
           Rectifier = {
             return("ReLU")
           },
           RectifierDropout = {
             return("ReLU")
           },
           Maxout = {
             stop("SensAnalysisMLP function is not ready for Maxout layers")
           },
           MaxoutDropout = {
             stop("SensAnalysisMLP function is not ready for Maxout layers")
           },
           Softmax = {
             return("sigmoid")
           },
           {
             stop("SensAnalysisMLP is not ready for the activation function used")
           })
  }
  actfun <- sapply(MLP.fit@model$model_summary$type, PrepActFuncs)
  # Call to the default function
  SensAnalysisMLP.default(finalModel,
                          trData = trData,
                          actfunc = actfun,
                          .returnSens = .returnSens,
                          .rawSens = .rawSens,
                          sens_origin_layer = sens_origin_layer,
                          sens_end_layer = sens_end_layer,
                          sens_origin_input = sens_origin_input,
                          sens_end_input = sens_end_input,
                          preProc = preProc,
                          terms = NULL,
                          plot = plot,
                          output_name = if("output_name" %in% names(args)){args$output_name}else{MLP.fit@parameters$y},
                          deractfunc = if("deractfunc" %in% names(args)){args$deractfunc}else{NULL},
                          args[!names(args) %in% c("trData","output_name","deractfunc")])
  }


#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP list
SensAnalysisMLP.list <- function(MLP.fit,
                                 .returnSens = TRUE,
                                 plot = TRUE,
                                 .rawSens = FALSE,
                                 sens_origin_layer = 1,
                                 sens_end_layer = "last",
                                 sens_origin_input = TRUE,
                                 sens_end_input = FALSE,
                                 trData,
                                 actfunc,
                                 ...) {
  # For a neural nnet
  args <- list(...)
  ## Detect that it's from the neural package
  neuralfields <- c("weight","dist", "neurons", "actfns", "diffact")
  if (!all(neuralfields %in% names(MLP.fit))){
    stop("Object detected is not an accepted list object")
  }
  finalModel <- NULL
  finalModel$n <- MLP.fit$neurons
  # Try to extract the weights of the neural network
  wts <- c()
  for (i in 1:(length(c(finalModel$n))-1)) {
    for (j in 1:finalModel$n[i+1]) {
      wtsi <- MLP.fit$weight[[i]][,j]
      bsi <- MLP.fit$dist[[i+1]][j]
      wts <- c(wts, bsi, wtsi)
    }
  }
  finalModel$wts <- wts
  if ("output_name" %in% names(args)) {
    finalModel$coefnames <- names(trData)[names(trData) != args$output_name]
  } else {
    finalModel$coefnames <- names(trData)[names(trData) != ".outcome"]
  }

  # By default, the activation functions is linear, sigmoid, sigmoid
  if (is.null(actfunc)) {
    actfunc <- c("linear","sigmoid","sigmoid")
  } else {
    # See ?neural::mlptrain to see which number correspond to which activation function
    actfunc <- c("linear",
                 ifelse(actfunc == 1, "sigmoid",
                   ifelse(actfunc == 2, "tanh",
                     ifelse(actfunc == 3, "Gauss",
                       ifelse(actfunc == 4, "linear"
                         )))))
  }

  SensAnalysisMLP.default(finalModel,
                          trData = trData,
                          .returnSens = .returnSens,
                          .rawSens = .rawSens,
                          sens_origin_layer = sens_origin_layer,
                          sens_end_layer = sens_end_layer,
                          sens_origin_input = sens_origin_input,
                          sens_end_input = sens_end_input,
                          actfunc = actfunc,
                          preProc = NULL,
                          terms = NULL,
                          plot = plot,
                          output_name = if("output_name" %in% names(args)){args$output_name}else{".outcome"},
                          deractfunc = if("deractfunc" %in% names(args)){args$deractfunc}else{NULL},
                          args[!names(args) %in% c("output_name","deractfunc")])
}

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP mlp
SensAnalysisMLP.mlp <- function(MLP.fit,
                                .returnSens = TRUE,
                                plot = TRUE,
                                .rawSens = FALSE,
                                sens_origin_layer = 1,
                                sens_end_layer = "last",
                                sens_origin_input = TRUE,
                                sens_end_input = FALSE,
                                trData,
                                preProc = NULL,
                                terms = NULL,
                                ...) {
  args <- list(...)
  # For a RSNNS mlp
  netInfo <- RSNNS::extractNetInfo(MLP.fit)
  nwts <- NeuralNetTools::neuralweights(MLP.fit)
  finalModel <- NULL
  finalModel$n <- nwts$struct
  # Fill NAs with the corresponding bias
  for (i in netInfo$unitDefinitions$unitNo[netInfo$unitDefinitions$type %in%
                                           c("UNIT_HIDDEN")]) {
    hidname <- paste("hidden",
                     as.character(netInfo$unitDefinitions$posY[i]/2),
                     as.character(netInfo$unitDefinitions$posX[i]),
                     sep = " ")
    nwts$wts[[which(hidname == names(nwts$wts))]][1] <- netInfo$unitDefinitions$unitBias[i]
  }
  for (i in netInfo$unitDefinitions$unitNo[netInfo$unitDefinitions$type %in%
                                           c("UNIT_OUTPUT")]) {
    outname <- paste("out",
                     as.character(substr(netInfo$unitDefinitions$unitName[i],
                                         nchar(netInfo$unitDefinitions$unitName[i]),
                                         nchar(netInfo$unitDefinitions$unitName[i]))),
                     sep = " ")
    nwts$wts[[which(outname == names(nwts$wts))]][1] <- netInfo$unitDefinitions$unitBias[i]
  }

  wts <- c()
  for (i in 1:length(nwts$wts)){
    wts <- c(wts, nwts$wts[[i]])
  }

  wts[is.na(wts)] <- 0
  finalModel$wts <- wts
  finalModel$coefnames <- substr(netInfo$unitDefinitions$unitName[1:finalModel$n[1]],
                                 nchar("Input_")+1, 100000)
  PrepActFun <- function(acfun) {
    # Switch case to define which value it returns
    switch(acfun,
           Act_Identity = {
             return("linear")
           },
           Act_TanH = {
             return("tanh")
           },
           Act_StepFunc = {
             return("step")
           },
           Act_Logistic = {
             return("sigmoid")
           },
           {
             stop("SensAnalysisMLP is not ready for the activation function used")
           })
  }
  actfun <- sapply(unique(cbind(substr(netInfo$unitDefinitions$unitName,1,5),
                                netInfo$unitDefinitions$actFunc))[,2], PrepActFun)
  if(length(actfun) != length(finalModel$n)) {
    # This is done in case of several hidden layers with same activation function
    actfun <- c(actfun[1], rep(actfun[2], length(finalModel$n)-2),actfun[length(actfun)])
  }

  SensAnalysisMLP.default(finalModel,
                          trData = trData,
                          actfunc = actfun,
                          .returnSens = .returnSens,
                          .rawSens = .rawSens,
                          sens_origin_layer = sens_origin_layer,
                          sens_end_layer = sens_end_layer,
                          sens_origin_input = sens_origin_input,
                          sens_end_input = sens_end_input,
                          preProc = preProc,
                          terms = terms,
                          plot = plot,
                          output_name = if("output_name" %in% names(args)){args$output_name}else{".outcome"},
                          deractfunc = if("deractfunc" %in% names(args)){args$deractfunc}else{NULL},
                          args[!names(args) %in% c("output_name","deractfunc")])
}

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP nn
SensAnalysisMLP.nn <- function(MLP.fit,
                               .returnSens = TRUE,
                               plot = TRUE,
                               .rawSens = FALSE,
                               sens_origin_layer = 1,
                               sens_end_layer = "last",
                               sens_origin_input = TRUE,
                               sens_end_input = FALSE,
                               preProc = NULL,
                               terms = NULL,
                               ...) {

  # For a neuralnet nn
  args <- list(...)
  finalModel <- NULL
  finalModel$coefnames <- MLP.fit$model.list$variables
  trData <- MLP.fit$data
  actfun <- c("linear",
              rep(ifelse(attributes(MLP.fit$act.fct)$type == "tanh", "tanh", "sigmoid"),
                  length(MLP.fit$weights[[1]])-1),
              ifelse(MLP.fit$linear.output, "linear", "sigmoid"))
  sensit <- array(NA, dim = c(length(MLP.fit$weights)*nrow(trData),
                              length(MLP.fit$model.list$variables),
                              length(MLP.fit$model.list$response)))
  for (j in 1:length(MLP.fit$weights)) {
    finalModel$n <- NULL
    wts <- c()
    for (i in 1:length(MLP.fit$weights[[j]])) {
      wts <- c(wts, as.vector(MLP.fit$weights[[j]][[i]]))
      finalModel$n <- c(finalModel$n, dim(MLP.fit$weights[[j]][[i]])[1]-1)
    }
    finalModel$n <- c(finalModel$n, dim(MLP.fit$weights[[j]][[i]])[2])
    finalModel$wts <- wts
    sensitivities <- SensAnalysisMLP.default(finalModel,
                            trData = trData,
                            actfunc = actfun,
                            .returnSens = TRUE,
                            .rawSens = TRUE,
                            sens_origin_layer = sens_origin_layer,
                            sens_end_layer = sens_end_layer,
                            sens_origin_input = sens_origin_input,
                            sens_end_input = sens_end_input,
                            preProc = preProc,
                            terms = terms,
                            plot = FALSE,
                            output_name = names(trData)[names(trData) == MLP.fit$model.list$response],
                            deractfunc = if("deractfunc" %in% names(args)){args$deractfunc}else{NULL},
                            args[!names(args) %in% c("deractfunc")])
    sensit[((j-1)*nrow(trData)+1):(j*nrow(trData)),,] <- sensitivities
  }
  colnames(sensit) <- finalModel$coefnames

  sens <-
    data.frame(
      varNames = finalModel$coefnames,
      mean = colMeans(sensit[, , 1], na.rm = TRUE),
      std = apply(sensit[, , 1], 2, stats::sd, na.rm = TRUE),
      meanSensSQ = colMeans(sensit[, , 1] ^ 2, na.rm = TRUE)
    )

  if (plot) {
    # show plots if required
    NeuralSens::SensitivityPlots(sens,der = sensit[,,1])
  }

  if (.returnSens) {
    if(!.rawSens) {
      # Check if there are more than one output
      return(sens)
    } else {
      # Return sensitivities without processing
      return(sensit)
    }
  }
}

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP nnet
SensAnalysisMLP.nnet <- function(MLP.fit,
                                 .returnSens = TRUE,
                                 plot = TRUE,
                                 .rawSens = FALSE,
                                 sens_origin_layer = 1,
                                 sens_end_layer = "last",
                                 sens_origin_input = TRUE,
                                 sens_end_input = FALSE,
                                 trData,
                                 preProc = NULL,
                                 terms = NULL,
                                 ...) {
  # For a nnet nnet
  args <- list(...)
  # Check if some arguments has been changed in a parent function
  if(length(args) == 1 &&  is.list(args[[1]])) {
    args <- as.list(...)
  }
  finalModel <- NULL
  finalModel$n <- MLP.fit$n
  finalModel$wts <- MLP.fit$wts
  finalModel$coefnames <- MLP.fit$coefnames
  if(!any(names(trData) == ".outcome")){
    if (!"output_name" %in% names(args)) {
      names(trData)[!names(trData) %in% attr(MLP.fit$terms,"term.labels")] <- ".outcome"
    }
  }

  actfun <- c("linear","sigmoid",
              ifelse(is.factor(trData$.outcome),"sigmoid","linear"))
  SensAnalysisMLP.default(finalModel,
                          trData = trData,
                          actfunc = actfun,
                          .returnSens = .returnSens,
                          sens_origin_layer = sens_origin_layer,
                          sens_end_layer = sens_end_layer,
                          sens_origin_input = sens_origin_input,
                          sens_end_input = sens_end_input,
                          .rawSens = .rawSens,
                          preProc = preProc,
                          terms = terms,
                          plot = plot,
                          output_name = if("output_name" %in% names(args)){args$output_name}else{".outcome"},
                          deractfunc = if("deractfunc" %in% names(args)){args$deractfunc}else{NULL},
                          args[!names(args) %in% c("output_name","deractfunc")])
}

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP nnetar
SensAnalysisMLP.nnetar <- function(MLP.fit,
                                   .returnSens = TRUE,
                                   plot = TRUE,
                                   .rawSens = FALSE,
                                   sens_origin_layer = 1,
                                   sens_end_layer = "last",
                                   sens_origin_input = TRUE,
                                   sens_end_input = FALSE,
                                   ...) {
  # Create the lags in the trData
  args = list(...)
  if (!is.null(MLP.fit$xreg)) {
    if ("trData" %in% names(args)) {
      xreg <- as.data.frame(args$trData[,attr(MLP.fit$xreg,"dimnames")[[2]]])
      if(".outcome" %in% names(args$trData)) {
        outcome <- args$trData$.outcome
      } else if ("output_name" %in% names(args)) {
        outcome <- args$trData[,args$output_name]
      } else {
        stop("Change the name of the output variable to '.outcome' or
             provide the name of the output using the 'output_name' argument")
      }
    } else {
      xreg <- as.data.frame(MLP.fit$xreg)[MLP.fit$subset,]
      outcome <- MLP.fit$x[MLP.fit$subset]
    }
  } else {
    outcome <- MLP.fit$x[MLP.fit$subset]
  }
  # Scale the regressors
  if (!is.null(MLP.fit$scalexreg)) {
    for (i in 1:ncol(xreg)) {
      varname <- names(xreg)[[i]]
      indexscale <- which(attr(MLP.fit$scalexreg$center,"names") == varname)
      xreg[[i]] <- (xreg[[i]] - MLP.fit$scalexreg$center[indexscale])/MLP.fit$scalexreg$scale[indexscale]
    }
  }
  # Scale the output
  if (!is.null(MLP.fit$scalex)) {
    outcome <- (outcome - MLP.fit$scalex$center)/MLP.fit$scalex$scale
  }

  # Create lagged outcome as input
  ylagged <- NULL
  p <- MLP.fit$p
  P <- MLP.fit$P
  m <- MLP.fit$m
  if (P > 0) {
    lags <- sort(unique(c(1:p, m * (1:P))))
  } else {
    lags <- 1:p
  }
  # Create name for the lags
  if ("output_name" %in% names(args)) {
    out_nameLag <- paste0(".",args$output_name,"_Lag")
  } else {
    out_nameLag <- ".outcome_Lag"
  }
  for (i in lags) {
    ylagged[[i]] <- Hmisc::Lag(outcome, i)
    names(ylagged)[i] <- paste0(out_nameLag, as.character(i))
  }
  ylagged <- as.data.frame(ylagged[lags])

  if (!is.null(MLP.fit$xreg)) {
    trData <- cbind(ylagged, xreg, as.data.frame(outcome))
    varnames <- c(names(trData)[1:length(lags)],names(as.data.frame(MLP.fit$xreg)))
  } else {
    trData <- cbind(ylagged, as.data.frame(outcome))
    varnames <- names(trData)[1:length(lags)]
  }

  if ("output_name" %in% names(args)) {
    names(trData)[ncol(trData)] <- args$output_name
  } else {
    names(trData)[ncol(trData)] <- ".outcome"
  }

  # Get rid of rows with NAs
  trData <- trData[stats::complete.cases(trData),]

  # For a nnet nnet
  finalModel <- NULL
  sensitivities <- list()
  finalModel$n <- MLP.fit$model[[1]]$n
  actfun <- c("linear","sigmoid",
              ifelse(is.factor(trData$.outcome),"sigmoid","linear"))
  finalModel$coefnames <- varnames
  # Apply default function to all the models in the nnetar object
  sensit <- array(NA, dim = c(length(MLP.fit$model)*nrow(trData),
                              MLP.fit$model[[1]]$n[1],
                              MLP.fit$model[[1]]$n[length(MLP.fit$model[[1]]$n)]))
  for (i in 1:length(MLP.fit$model)) {
    finalModel$wts <- MLP.fit$model[[1]]$wts
    sensitivities[[i]] <-  SensAnalysisMLP.default(finalModel,
                                              trData = trData,
                                              actfunc = actfun,
                                              deractfunc = if("deractfunc" %in% names(args)){args$deractfunc}else{NULL},
                                              .returnSens = TRUE,
                                              .rawSens = TRUE,
                                              sens_origin_layer = sens_origin_layer,
                                              sens_end_layer = sens_end_layer,
                                              sens_origin_input = sens_origin_input,
                                              sens_end_input = sens_end_input,
                                              preProc = NULL,
                                              terms = NULL,
                                              plot = FALSE,
                                              output_name = if("output_name" %in% names(args)){args$output_name}else{".outcome"})
    sensit[((i-1)*nrow(trData)+1):(i*nrow(trData)),,] <- sensitivities[[i]]
  }

  colnames(sensit) <- finalModel$coefnames

  sens <-
    data.frame(
      varNames = varnames,
      mean = colMeans(sensit[, , 1], na.rm = TRUE),
      std = apply(sensit[, , 1], 2, stats::sd, na.rm = TRUE),
      meanSensSQ = colMeans(sensit[, , 1] ^ 2, na.rm = TRUE)
    )

  if (plot) {
    # show plots if required
    NeuralSens::SensitivityPlots(sens,der = sensit[,,1])
  }

  if (.returnSens) {
    if(!.rawSens) {
      # Check if there are more than one output
      return(sens)
    } else {
      # Return sensitivities without processing
      return(sensit)
    }
  }
}

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP numeric
SensAnalysisMLP.numeric <- function(MLP.fit,
                                    .returnSens = TRUE,
                                    plot = TRUE,
                                    .rawSens = FALSE,
                                    sens_origin_layer = 1,
                                    sens_end_layer = "last",
                                    sens_origin_input = TRUE,
                                    sens_end_input = FALSE,
                                    trData,
                                    actfunc = NULL,
                                    preProc = NULL,
                                    terms = NULL,
                                    ...) {
  # Generic method when the weights are passed in the argument MLP.fit
  finalModel <- NULL
  finalModel$wts <- MLP.fit
  # The mlp structure and the name of the explanatory variables should be passed
  # as mlpstr and coefnames argument respectively
  args <- list(...)
  if (!"mlpstr" %in% names(args)) {
    stop("MLP structure must be passed in mlpstr argument")
  }
  finalModel$n <- args$mlpstr
  # Define the names of the explanatory variables
  if ((".outcome" %in% names(trData)) || "output_name" %in% names(args)) {
    if (!("coefnames" %in% names(args))) {
      if ("output_name" %in% names(args)) {
        finalModel$coefnames <- names(trData)[names(trData) != args$output_name]
      } else {
        finalModel$coefnames <- names(trData)[names(trData) != ".outcome"]
      }
    } else {
      finalModel$coefnames <- args$coefnames
    }
  } else {
    if (!("coefnames" %in% names(args))) {
      stop("Names of explanatory variables must be passed in coefnames argument")
    }
    finalModel$coefnames <- args$coefnames
    if (!all(args$coefnames %in% names(trData))) {
      stop("Explanatory variables defined in coefnames has not been found in trData")
    }
  }
  # Define the activation functions used in the neural network
  # The activation functions must be passed as actfun argument
  if (length(actfunc) != length(args$mlpstr)) {
    stop("Number of activation functions does not match the structure of the MLP")
  }

  SensAnalysisMLP.default(finalModel,
                          trData = trData,
                          actfunc = actfunc,
                          .returnSens = .returnSens,
                          .rawSens = .rawSens,
                          sens_origin_layer = sens_origin_layer,
                          sens_end_layer = sens_end_layer,
                          sens_origin_input = sens_origin_input,
                          sens_end_input = sens_end_input,
                          preProc = preProc,
                          terms = terms,
                          plot = plot,
                          output_name = if("output_name" %in% names(args)){args$output_name}else{".outcome"},
                          deractfunc = if("deractfunc" %in% names(args)){args$deractfunc}else{NULL},
                          args[!names(args) %in% c("output_name","deractfunc")])
}
