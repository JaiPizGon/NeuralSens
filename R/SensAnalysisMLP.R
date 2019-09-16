#' Sensitivity of NNET models
#'
#' @description Function for evaluating the sensitivities of the inputs
#'   variables in a mlp model
#' @param MLP.fit fitted model from caret package using nnet method
#' @param trData data frame containing the training data of the model
#' @param actfunc character vector indicating the activation function of each
#'   neurons layer.
#' @param .returnSens logical value. If \code{TRUE}, sensibility of the model is
#'   returned.
#' @param preProc preProcess structure applied to the training data. See also
#'   \code{\link[caret]{preProcess}}
#' @param terms function applied to the training data to create factors. See
#'   also \code{\link[caret]{train}}
#' @param plot \code{logical} whether or not to plot the analysis. By default
#'   is \code{TRUE}.
#' @param .rawSens \code{logical} whether or not to return the sensitivity of each row
#' of the data provided, or return the mean, sd and mean of the square of the
#' sensitivities. By default is \code{FALSE}.
#' @param ...	additional arguments passed to or from other methods
#' @return dataframe with the sensitivities obtained for each variable if
#'   \code{.returnSens = TRUE}. If \code{.returnSens = FALSE}, the sensitivities without
#'   processing are returned in a 3D array. If there is more than one output, the
#'   sensitivities of each output are given in a list.
#' @section Plots: \itemize{ \item Plot 1: colorful plot with the
#'   classification of the classes in a 2D map \item Plot 2: b/w plot with
#'   probability of the chosen class in a 2D map \item Plot 3: plot with the
#'   stats::predictions of the data provided }
#' @details In case of using an input of class \code{factor} and a package which
#'   need to enter the input data as matrix, the dummies must be created before
#'   training the neural network.
#'
#'   After that, the training data must be given to the function using the
#'   \code{trData} argument.
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
#' fdata.Reg.tr[,2:3] <- fdata.Reg.tr[,2:3]/10
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
#' NeuralSens::SensAnalysisMLP(nnetmod, trData = nntrData)
#'
#' \donttest{
#'
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
#' # Creaci?n de un cluster local con todos los cores disponibles
#' h2o::h2o.init(ip = "localhost",
#'               # -1 indica que se empleen todos los cores disponibles.
#'               nthreads = 4,
#'               # M?xima memoria disponible para el cluster.
#'               max_mem_size = "2g")
#'
#' # Se eliminan los datos del cluster por si ya hab?a sido iniciado.
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
#' # Apaga el cluster
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
#' names(trData)[1] <- ".outcome"
#' NeuralSens::SensAnalysisMLP(neuralmod, trData = trData)
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
#' names(trData)[1] <- ".outcome"
#'
#' # Try SensAnalysisMLP
#' NeuralSens::SensAnalysisMLP(RSNNSmod, trData = trData)
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
#' # Creaci?n de un cluster local con todos los cores disponibles
#' h2o::h2o.init(ip = "localhost",
#'               # -1 indica que se empleen todos los cores disponibles.
#'               nthreads = 4,
#'               # M?xima memoria disponible para el cluster.
#'               max_mem_size = "2g")
#'
#' # Se eliminan los datos del cluster por si ya hab?a sido iniciado.
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
#' # names(trData)[1] <- ".outcome"
#' #
#' # # Try SensAnalysisMLP
#' # NeuralSens::SensAnalysisMLP(RSNNSmod, trData = trData)
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
SensAnalysisMLP <- function(MLP.fit, .returnSens = TRUE, plot = TRUE, .rawSens = FALSE, ...) UseMethod('SensAnalysisMLP')

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP default
SensAnalysisMLP.default <- function(MLP.fit, .returnSens = TRUE, plot = TRUE, .rawSens = FALSE, trData,
                                    actfunc = NULL,preProc = NULL,
                                    terms = NULL, ...) {
  ### Things needed for calculating the sensibilities:
  #   - Structure of the model  -> MLP.fit$n
  #   - Weights of the model    -> MLP.fit$wts
  #   - Name of the inputs      -> MLP.fit$coefnames
  #   - trData [output + inputs], output's name must be .outcome
  #   - modelType: "Regression" or "Classification"

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
  D <- list()
  W <- list()
  # Initialize the activation and the derivative of the activation function for each layer
  ActivationFunction <- lapply(actfunc, NeuralSens::ActFunc)
  DerActivationFunction <- lapply(actfunc, NeuralSens::DerActFunc)

  W[[1]] <- array(c(0,rep(1,ncol(TestData))),c(ncol(TestData)+1,1,nrow(TestData)))
  # For each row in the TestData
  Z[[1]] <- as.matrix(TestData)
  O[[1]] <- ActivationFunction[[1]](Z[[1]])
  # Build the jacobian of the first layer
  D[[1]] <- array(NA, dim=c(mlpstr[1], mlpstr[1], nrow(TestData)))
  for(irow in 1:nrow(TestData)){
    D[[1]][,,irow] <- DerActivationFunction[[1]](Z[[1]][irow,])
  }
  # For each layer, calculate the input to the activation functions of each layer
  # This inputs are gonna be used to calculate the derivatives and the output of each layer
  for (l in 2:length(mlpstr)){
    W[[l]] <- data.matrix(as.data.frame(wts[(sum(mlpstr[1:(l-1)])-mlpstr[1]+1):(sum(mlpstr[1:l])-mlpstr[1])]))
    Z[[l]] <- cbind(1, O[[l-1]]) %*% W[[l]]
    O[[l]] <- ActivationFunction[[l]](Z[[l]])

    # Detect if it's a vector because we need it in row vectors and in r a vector is a column
    m <- W[[l]][2:nrow(W[[l]]),]
    D[[l]]<- array(NA, dim=c(mlpstr[1], mlpstr[l], nrow(TestData)))
    # Chain rule for calculate the derivatives between layers
    for(irow in 1:nrow(TestData)){
      z <- DerActivationFunction[[l]](Z[[l]][irow,])
      D[[l]][,,irow] <- D[[l-1]][,,irow] %*% m %*% z
    }
  }
  # Output of the neural network is the output of the last layer
  out <- O[[length(O)]]
  der <- aperm(D[[l]],c(3,1,2))
  colnames(der) <- varnames
  # Obtain sensitivities of the first output and create plots if required
  sens <-
    data.frame(
      varNames = varnames,
      mean = colMeans(der[, , 1], na.rm = TRUE),
      std = apply(der[, , 1], 2, stats::sd, na.rm = TRUE),
      meanSensSQ = colMeans(der[, , 1] ^ 2, na.rm = TRUE)
    )

  if (plot) {
    # show plots if required
    NeuralSens::SensitivityPlots(sens,der = der[,,1])
  }

  if (.returnSens) {
    if(!.rawSens) {
      # Check if there are more than one output and return a list
      # with the sensitivities of each output. If not, return a data.frame
      if (dim(der)[3] > 1) {
        sens <- list(sens)
        for (i in 2:dim(der)[3]) {
          sens[[i]] <- data.frame(
            varNames = varnames,
            mean = colMeans(der[, , i], na.rm = TRUE),
            std = apply(der[, , i], 2, stats::sd, na.rm = TRUE),
            meanSensSQ = colMeans(der[, , i] ^ 2, na.rm = TRUE)
          )
        }
        names(sens) <- make.names(levels(trData$.outcome), unique = TRUE)
      }
      return(sens)
    } else {
      # Return sensitivities without processing
      return(der)
    }
  }
}

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP train
SensAnalysisMLP.train <- function(MLP.fit, .returnSens = TRUE, plot = TRUE, .rawSens = FALSE,...) {
  args <- list(...)
  SensAnalysisMLP(MLP.fit$finalModel,
                  trData = if ("trData" %in% names(args)) {args$trData} else {MLP.fit$trainingData},
                  .returnSens = .returnSens,
                  .rawSens = .rawSens,
                  preProc = if ("preProc" %in% names(args)) {args$preProc} else {MLP.fit$preProcess},
                  terms = if ("terms" %in% names(args)) {args$terms} else {MLP.fit$terms},
                  plot = plot, args[!names(args) %in% c("trData","preProc","terms")])
}

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP H2OMultinomialModel
SensAnalysisMLP.H2OMultinomialModel <- function(MLP.fit, .returnSens = TRUE, plot = TRUE, .rawSens = FALSE,...) {
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
                          preProc = preProc,
                          terms = NULL,
                          plot = plot, args[!names(args) %in% c("trData")])
  }

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP H2ORegressionModel
SensAnalysisMLP.H2ORegressionModel <- function(MLP.fit, .returnSens = TRUE, plot = TRUE, .rawSens = FALSE,...) {
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
                          preProc = preProc,
                          terms = NULL,
                          plot = plot, args[!names(args) %in% c("trData")])
  }


#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP list
SensAnalysisMLP.list <- function(MLP.fit, .returnSens = TRUE, plot = TRUE, .rawSens = FALSE, trData,...) {
  # For a neural nnet
  ## Detect that it's from the neural package
  neuralfields <- c("weight","dist", "neurons", "actfns", "diffact")
  if (!all(neuralfields %in% names(MLP.fit))){
    stop("Object detected is not from neural library")
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
  finalModel$coefnames <- names(trData)[names(trData) != ".outcome"]
  if (is.factor(trData$.outcome)) {
    modelType <- "Classification"
  } else {
    modelType <- "Regression"
  }
  SensAnalysisMLP.default(finalModel,
                          trData = trData,
                          modelType = modelType,
                          .returnSens = .returnSens,
                          .rawSens = .rawSens,
                          preProc = NULL,
                          terms = NULL,
                          plot = plot, ...)
}

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP mlp
SensAnalysisMLP.mlp <- function(MLP.fit, .returnSens = TRUE, plot = TRUE, .rawSens = FALSE, trData, preProc = NULL, terms = NULL, ...) {
  # For a RSNNS mlp
  netInfo <- RSNNS::extractNetInfo(MLP.fit)
  nwts <- NeuralNetTools::neuralweights(MLP.fit)
  finalModel <- NULL
  finalModel$n <- nwts$struct
  wts <- c()
  for (i in 1:length(nwts$wts)){
    wts <- c(wts, nwts$wts[[i]])
  }
  wts[is.na(wts)] <- netInfo$unitDefinitions$unitBias[(nrow(netInfo$unitDefinitions)-sum(finalModel$n[2:length(finalModel$n)])+1):nrow(netInfo$unitDefinitions)]
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
                          preProc = preProc,
                          terms = terms,
                          plot = plot, ...)
}

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP nn
SensAnalysisMLP.nn <- function(MLP.fit, .returnSens = TRUE, plot = TRUE, .rawSens = FALSE, preProc = NULL, terms = NULL, ...) {
  # For a neuralnet nn
  finalModel <- NULL
  finalModel$n <- c(nrow(MLP.fit$weights[[1]][[1]])-1,ncol(MLP.fit$weights[[1]][[1]]),ncol(MLP.fit$weights[[1]][[2]]))
  wts <- c()
  for(i in 1:ncol(MLP.fit$weights[[1]][[1]])){
    wts <- c(wts, MLP.fit$weights[[1]][[1]][,i])
  }
  for(i in 1:ncol(MLP.fit$weights[[1]][[2]])){
    wts <- c(wts, MLP.fit$weights[[1]][[2]][,i])
  }
  finalModel$wts <- wts
  finalModel$coefnames <- MLP.fit$model.list$variables
  trData <- MLP.fit$data
  names(trData)[names(trData) == MLP.fit$model.list$response] <- ".outcome"
  actfun <- c("linear",
              ifelse(attributes(MLP.fit$act.fct)$type == "tanh", "tanh", "sigmoid"),
              ifelse(MLP.fit$linear.output, "linear", "sigmoid"))
  SensAnalysisMLP.default(finalModel,
                          trData = trData,
                          actfunc = actfun,
                          .returnSens = .returnSens,
                          .rawSens = .rawSens,
                          preProc = preProc,
                          terms = terms,
                          plot = plot, ...)
}

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP nnet
SensAnalysisMLP.nnet <- function(MLP.fit, .returnSens = TRUE, plot = TRUE, .rawSens = FALSE, trData, preProc = NULL, terms = NULL, ...) {
  # For a nnet nnet
  finalModel <- NULL
  finalModel$n <- MLP.fit$n
  finalModel$wts <- MLP.fit$wts
  finalModel$coefnames <- MLP.fit$coefnames
  if(!any(names(trData) == ".outcome")){
    names(trData)[!names(trData) %in% attr(MLP.fit$terms,"term.labels")] <- ".outcome"
  }

  actfun <- c("linear","sigmoid",
              ifelse(is.factor(trData$.outcome),"sigmoid","linear"))
  SensAnalysisMLP.default(finalModel,
                          trData = trData,
                          actfunc = actfun,
                          .returnSens = .returnSens,
                          .rawSens = .rawSens,
                          preProc = preProc,
                          terms = terms,
                          plot = plot, ...)
}

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP nnetar
SensAnalysisMLP.nnetar <- function(MLP.fit, .returnSens = TRUE, plot = TRUE, .rawSens = FALSE,...) {
  # Create the lags in the trData
  args = list(...)
  if (!is.null(MLP.fit$xreg)) {
    if ("trData" %in% names(args)) {
      xreg <- args$trData[,attr(MLP.fit$xreg,"dimnames")[[2]]]
      if(!".outcome" %in% names(args$trData)) {
        stop("Change the name of the output variable to '.outcome'")
      }
      outcome <- args$trData$.outcome
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
  for (i in lags) {
    ylagged[[i]] <- Hmisc::Lag(outcome, i)
    names(ylagged)[i] <- paste0(".outcome_Lag",as.character(i))
  }
  ylagged <- as.data.frame(ylagged[lags])

  if (!is.null(MLP.fit$xreg)) {
    trData <- cbind(ylagged, xreg, as.data.frame(outcome))
    varnames <- c(names(trData)[1:MLP.fit$p],names(as.data.frame(MLP.fit$xreg)))
  } else {
    trData <- cbind(ylagged, as.data.frame(outcome))
    varnames <- names(trData)[1:MLP.fit$p]
  }

  names(trData)[ncol(trData)] <- ".outcome"
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
  for (i in 1:length(MLP.fit$model)) {
    finalModel$wts <- MLP.fit$model[[1]]$wts
    sensitivities[[i]] <-  SensAnalysisMLP.default(finalModel,
                                              trData = trData,
                                              actfunc = actfun,
                                              .returnSens = TRUE,
                                              .rawSens = TRUE,
                                              preProc = NULL,
                                              terms = NULL,
                                              plot = FALSE)
  }

  sensitivities <- as.data.frame(do.call("rbind",lapply(sensitivities,
                                               function(x) {
                                                 as.data.frame(x[1:dim(x)[1],1:dim(x)[2],1])
                                               })))

  sens <-
    data.frame(
      varNames = varnames,
      mean = base::colMeans(sensitivities, na.rm = TRUE),
      std = sapply(sensitivities,stats::sd, na.rm = TRUE),
      meanSensSQ = base::colMeans(sensitivities ^ 2, na.rm = TRUE)
    )

  if (plot) {
    plotlist <- list()

    plotlist[[1]] <- ggplot2::ggplot(sens) +
      ggplot2::geom_point(ggplot2::aes_string(x = "mean", y = "std")) +
      ggplot2::geom_label(ggplot2::aes_string(x = "mean", y = "std", label = "varnames"),
                          position = "nudge") +
      ggplot2::geom_point(ggplot2::aes(x = 0, y = 0), size = 5, color = "blue") +
      ggplot2::geom_hline(ggplot2::aes(yintercept = 0), color = "blue") +
      ggplot2::geom_vline(ggplot2::aes(xintercept = 0), color = "blue") +
      # coord_cartesian(xlim = c(min(sens$mean,0)-0.1*abs(min(sens$mean,0)), max(sens$mean)+0.1*abs(max(sens$mean))), ylim = c(0, max(sens$std)*1.1))+
      ggplot2::labs(x = "mean(Sens)", y = "std(Sens)")


    plotlist[[2]] <- ggplot2::ggplot() +
      ggplot2::geom_col(ggplot2::aes(x = varnames, y = colMeans(sensitivities[, , 1] ^ 2, na.rm = TRUE),
                                     fill = colMeans(sensitivities[, , 1] ^ 2, na.rm = TRUE))) +
      ggplot2::labs(x = "Input variables", y = "mean(Sens^2)") + ggplot2::guides(fill = "none")

    der2 <- as.data.frame(sensitivities[, , 1])
    colnames(der2) <- varnames
    dataplot <- reshape2::melt(der2, measure.vars = varnames)
    # bwidth <- sd(dataplot$value)/(1.34*(dim(dataplot)[1]/length(varnames)))
    # In case the data std is too narrow and erase the data
    if (any(abs(dataplot$value) > 2*max(sens$std, na.rm = TRUE)) ||
        max(abs(dataplot$value)) < max(sens$std, na.rm = TRUE)) {
      plotlist[[3]] <- ggplot2::ggplot(dataplot) +
        ggplot2::geom_density(ggplot2::aes_string(x = "value", fill = "variable"),
                              alpha = 0.4,
                              bw = "bcv") +
        ggplot2::labs(x = "Sens", y = "density(Sens)") +
        ggplot2::xlim(-1 * max(abs(dataplot$value), na.rm = TRUE),
                      1 * max(abs(dataplot$value), na.rm = TRUE))
    } else {
      plotlist[[3]] <- ggplot2::ggplot(dataplot) +
        ggplot2::geom_density(ggplot2::aes_string(x = "value", fill = "variable"),
                              alpha = 0.4,
                              bw = "bcv") +
        ggplot2::labs(x = "Sens", y = "density(Sens)") +
        ggplot2::xlim(-2 * max(sens$std, na.rm = TRUE), 2 * max(sens$std, na.rm = TRUE))
    }
    # Plot the list of plots created before
    gridExtra::grid.arrange(grobs = plotlist,
                            nrow  = length(plotlist),
                            ncols = 1)
  }

  if (.returnSens) {
    if(!.rawSens) {
      # Check if there are more than one output
      return(sens)
    } else {
      # Return sensitivities without processing
      colnames(sensitivities) <- finalModel$coefnames
      return(sensitivities)
    }
  }
}
