#' Sensitivity of NNET models
#'
#' @description Function for evaluating the sensitivities of the inputs variables in a mlp model
#' @param MLP.fit fitted model from caret package using nnet method
#' @param trData data frame containing the training data of the model
#' @param actfunc character vector indicating the activation function of each neurons layer.
#' @param .returnSens logical value. If \code{TRUE}, sensibility of the model is returned.
#' @param preProc preProcess structure applied to the training data
#' @param terms function applied to the training data to create factors
#' @param ...	additional arguments passed to or from other methods
#' @return dataframe with the sensitivities obtained for each variable if .returnSens \code{TRUE}
#' @section Output:
#' \itemize{
#'   \item Plot 1: colorful plot with the classification of the classes in a 2D map
#'   \item Plot 2: b/w plot with probability of the chosen class in a 2D map
#'   \item Plot 3: plot with the stats::predictions of the data provided
#' }
#' @examples
#' \dontrun{
#' ## Load necessary libraries ----------------------------------------------------
#' library(ggplot2)
#' library(caret)
#' library(h2o)
#' library(neural)
#' library(RSNNS)
#' library(neuralnet)
#'
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
#' ## Begin empty list of nnets ---------------------------------------------------
#' RegNNET <- NULL
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
#' ## Train caret NNET ------------------------------------------------------------
#' # Create trainControl
#' ctrl_tune <- trainControl(method = "boot",
#'                           savePredictions = FALSE,
#'                           summaryFunction = defaultSummary)
#' set.seed(150) #For replication
#' RegNNET$caret <- caret::train(form = DEM~.,
#'                               data = fdata.Reg.tr,
#'                               method = "nnet",
#'                               linout = TRUE,
#'                               tuneGrid = data.frame(size = hidden_neurons,
#'                                                     decay = decay),
#'                               maxit = iters,
#'                               preProcess = c("center","scale"),
#'                               trControl = ctrl_tune,
#'                               metric = "RMSE")
#'
#' # Try SensAnalysisMLP
#' NeuralSens::SensAnalysisMLP(RegNNET$caret)
#'
#' ## Train neural NNET -----------------------------------------------------------
#' set.seed(150)
#' RegNNET$neural <- mlptrain(as.matrix(nntrData[,2:ncol(nntrData)]),
#'                            hidden_neurons,
#'                            as.matrix(nntrData[1]),
#'                            it=iters,
#'                            visual=FALSE)
#'
#' # Try SensAnalysisMLP
#' trData <- nntrData
#' names(trData)[1] <- ".outcome"
#' NeuralSens::SensAnalysisMLP(RegNNET$neural, trData = trData)
#'
#' ## Train RSNNS NNET ------------------------------------------------------------
#' # Normalize data using RSNNS algorithms
#' trData <- as.data.frame(RSNNS::normalizeData(fdata.Reg.tr))
#' names(trData) <- names(fdata.Reg.tr)
#' set.seed(150)
#' RegNNET$RSNNS <- mlp(x = trData[,2:ncol(trData)],
#'                      y = trData[,1],
#'                      size = hidden_neurons,
#'                      linOut = TRUE,
#'                      learnFuncParams=c(decay),
#'                      maxit=iters)
#' names(trData)[1] <- ".outcome"
#'
#' # Try SensAnalysisMLP
#' NeuralSens::SensAnalysisMLP(RegNNET$RSNNS, trData = trData)
#'
#' ## TRAIN neuralnet NNET --------------------------------------------------------
#' # Create a formula to train NNET
#' form <- paste(names(fdata.Reg.tr)[2:ncol(fdata.Reg.tr)], collapse = " + ")
#' form <- formula(paste(names(fdata.Reg.tr)[1], form, sep = " ~ "))
#'
#' set.seed(150)
#' RegNNET$nn <- neuralnet(form,
#'                         nntrData,
#'                         linear.output = TRUE,
#'                         rep = 1,
#'                         hidden = hidden_neurons,
#'                         lifesign = "minimal",
#'                         threshold = 7,
#'                         stepmax = iters,
#'                         learningrate = decay,
#'                         act.fct = "tanh")
#'
#' # Try SensAnalysisMLP
#' NeuralSens::SensAnalysisMLP(RegNNET$nn)
#'
#' ################################################################################
#' #########################  CLASSIFICATION NNET #################################
#' ################################################################################
#' ## Begin empty list of nnets ---------------------------------------------------
#' ClassNNET <- NULL
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
#' ctrl_tune <- trainControl(method = "boot",
#'                           savePredictions = FALSE,
#'                           summaryFunction = defaultSummary)
#' set.seed(150) #For replication
#' ClassNNET$caret <- caret::train(form = DEM~.,
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
#' NeuralSens::SensAnalysisMLP(ClassNNET$caret)
#' ## Train neural NNET -----------------------------------------------------------
#' # set.seed(150)
#' # ClassNNET$neural <-mlptrain(as.matrix(nntrData[,2:ncol(nntrData)]),
#' #                           hidden_neurons,
#' #                           as.matrix(nntrData[1]),
#' #                           it=iters,
#' #                           visual=FALSE)
#' #
#' # # Try SensAnalysisMLP
#' # NeuralSens::SensAnalysisMLP(ClassNNET$neural, trData = trData)
#'
#' ### Train RSNNS NNET ------------------------------------------------------------
#' ## Normalize data using RSNNS algorithms
#' #trData <- as.data.frame(RSNNS::normalizeData(fdata.Reg.cl))
#' #names(trData) <- names(fdata.Reg.tr)
#' #set.seed(150)
#' #ClassNNET$RSNNS <- mlp(x = trData[,2:ncol(trData)],
#' #                       y = trData[,1],
#' #                       size = hidden_neurons,
#' #                       linOut = FALSE,
#' #                       learnFuncParams=c(decay),
#' #                       maxit=iters)
#' #names(trData)[1] <- ".outcome"
#'
#' ## Try SensAnalysisMLP
#' #NeuralSens::SensAnalysisMLP(ClassNNET$RSNNS, trData = trData)
#'
#' ## TRAIN neuralnet NNET --------------------------------------------------------
#' # Create a formula to train NNET
#' # form <- paste(names(fdata.Reg.tr)[2:ncol(fdata.Reg.tr)], collapse = " + ")
#' # form <- formula(paste(names(fdata.Reg.tr)[1], form, sep = " ~ "))
#' #
#' # set.seed(150)
#' # ClassNNET$nn <- neuralnet(form,
#' #                         nntrData,
#' #                         linear.output = FALSE,
#' #                         rep = 1,
#' #                         hidden = hidden_neurons,
#' #                         lifesign = "minimal",
#' #                         threshold = 4,
#' #                         stepmax = iters,
#' #                         learningrate = decay,
#' #                         act.fct = "tanh")
#' #
#' # # Try SensAnalysisMLP
#' # NeuralSens::SensAnalysisMLP(ClassNNET$nn)
#' ## Train h2o NNET --------------------------------------------------------------
#' # Creaci?n de un cluster local con todos los cores disponibles
#' h2o.init(ip = "localhost",
#'          # -1 indica que se empleen todos los cores disponibles.
#'          nthreads = 4,
#'          # M?xima memoria disponible para el cluster.
#'          max_mem_size = "2g")
#'
#' # Se eliminan los datos del cluster por si ya hab?a sido iniciado.
#' h2o.removeAll()
#' fdata_h2o <- as.h2o(x = fdata.Reg.tr, destination_frame = "fdata_h2o")
#'
#' set.seed(150)
#' RegNNET$h2o <- h2o.deeplearning(x = names(fdata.Reg.tr)[2:ncol(fdata.Reg.tr)],
#'                                 y = names(fdata.Reg.tr)[1],
#'                                 distribution = "AUTO",
#'                                 training_frame = fdata_h2o,
#'                                 standardize = TRUE,
#'                                 activation = "Tanh",
#'                                 hidden = c(hidden_neurons),
#'                                 stopping_rounds = 0,
#'                                 epochs = iters,
#'                                 seed = 150,
#'                                 model_id = "nnet_h2o",
#'                                 adaptive_rate = FALSE,
#'                                 rate_decay = decay,
#'                                 export_weights_and_biases = TRUE)
#'
#' # Try SensAnalysisMLP
#' NeuralSens::SensAnalysisMLP(RegNNET$h2o)
#'
#' # Apaga el cluster
#' h2o.shutdown(prompt = FALSE)
#' rm(fdata_h2o)
#'
#' ## Train h2o NNET --------------------------------------------------------------
#' # Creaci?n de un cluster local con todos los cores disponibles
#' h2o.init(ip = "localhost",
#'          # -1 indica que se empleen todos los cores disponibles.
#'          nthreads = 4,
#'          # M?xima memoria disponible para el cluster.
#'          max_mem_size = "2g")
#'
#' # Se eliminan los datos del cluster por si ya hab?a sido iniciado.
#' h2o.removeAll()
#' fdata_h2o <- as.h2o(x = fdata.Reg.cl, destination_frame = "fdata_h2o")
#'
#' set.seed(150)
#' ClassNNET$h2o <- h2o.deeplearning(x = names(fdata.Reg.cl)[2:ncol(fdata.Reg.cl)],
#'                                   y = names(fdata.Reg.cl)[1],
#'                                   distribution = "AUTO",
#'                                   training_frame = fdata_h2o,
#'                                   standardize = TRUE,
#'                                   activation = "Tanh",
#'                                   hidden = c(hidden_neurons),
#'                                   stopping_rounds = 0,
#'                                   epochs = iters,
#'                                   seed = 150,
#'                                   model_id = "nnet_h2o",
#'                                   adaptive_rate = FALSE,
#'                                   rate_decay = decay,
#'                                   export_weights_and_biases = TRUE)
#'
#' # Try SensAnalysisMLP
#' NeuralSens::SensAnalysisMLP(ClassNNET$h2o)
#'
#' # Apaga el cluster
#' h2o.shutdown(prompt = FALSE)
#' rm(fdata_h2o)
#'
#' }
#' @export
#' @rdname SensAnalysisMLP
SensAnalysisMLP <- function(MLP.fit, .returnSens = TRUE, ...) UseMethod('SensAnalysisMLP')

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP default
SensAnalysisMLP.default <- function(MLP.fit, .returnSens = TRUE, trData,
                                    actfunc = c('linear', 'sigmoid','linear'),preProc = NULL,
                                    terms = NULL, ...) {
  ### Things needed for calculating the sensibilities:
  #   - Structure of the model  -> MLP.fit$n
  #   - Weights of the model    -> MLP.fit$wts
  #   - Name of the inputs      -> MLP.fit$coefnames
  #   - trData [output + inputs], output's name must be .outcome
  #   - modelType: "Regression" or "Classification"
  options(warn = -1)

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

  # Output structure with the partial derivatives of the output by the input
  # and the output value for all the input data
  der <- array(rep(0, dim(TestData)[1]*dim(TestData)[2]*mlpstr[length(mlpstr)]),
              dim = c(dim(TestData)[1],dim(TestData)[2],mlpstr[length(mlpstr)]))
  out <- array(rep(0, dim(TestData)[1]*mlpstr[length(mlpstr)]),
               dim = c(dim(TestData)[1],mlpstr[length(mlpstr)]))
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
  # Build the jacobian of the first layer
  D[[1]] <- diag(ncol(TestData))
  W[[1]] <- matrix(c(0,rep(1,ncol(TestData))),nrow = ncol(TestData)+1, ncol = 1)
  # For each row in the TestData
  for (irow in 1:nrow(TestData)) {
    Z[[1]] <- as.numeric(TestData[irow,])
    O[[1]] <- sapply(as.numeric(TestData[irow,]),ActivationFunction[[1]])
    # Z[[1]] <- as.matrix.data.frame(TestData)
    # O[[1]] <- sapply(TestData, function(x){sapply(x,ActivationFunction[[1]])})

    # For each layer, calculate the input to the activation functions of each layer
    # This inputs are gonna be used to calculate the derivatives and the output of each layer
    for (l in 2:length(mlpstr)){
      W[[l]] <- data.matrix(as.data.frame(wts[(sum(mlpstr[1:(l-1)])-mlpstr[1]+1):(sum(mlpstr[1:l])-mlpstr[1])]))
      Z[[l]] <- as.vector(c(1, O[[l-1]]) %*% W[[l]])
      O[[l]] <- sapply(Z[[l]],ActivationFunction[[l]])
      D[[l]] <- t(W[[l]][2:nrow(W[[l]]),] *
                    sapply(Z[[l-1]], DerActivationFunction[[l-1]])) %*% D[[l-1]]
      }
    # Output of the neural network is the output of the last layer
    out[irow] <- O[[length(O)]]
    der[irow,,] <- (D[[length(D)]] * sapply(Z[[length(Z)]], DerActivationFunction[[length(Z)]]))[1,]
  }

  sens <-
    data.frame(
      varNames = varnames,
      mean = colMeans(der[, , 1], na.rm = TRUE),
      std = apply(der[, , 1], 2, stats::sd, na.rm = TRUE),
      meanSensSQ = colMeans(der[, , 1] ^ 2, na.rm = TRUE)
    )


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
    ggplot2::geom_col(ggplot2::aes(x = varnames, y = colMeans(der[, , 1] ^ 2, na.rm = TRUE),
                                  fill = colMeans(der[, , 1] ^ 2, na.rm = TRUE))) +
    ggplot2::labs(x = "Input variables", y = "mean(Sens^2)") + ggplot2::guides(fill = "none")

  der2 <- as.data.frame(der[, , 1])
  colnames(der2) <- varnames
  dataplot <- reshape2::melt(der2, measure.vars = varnames)
  # bwidth <- sd(dataplot$value)/(1.34*(dim(dataplot)[1]/length(varnames)))
  plotlist[[3]] <- ggplot2::ggplot(dataplot) +
    ggplot2::geom_density(ggplot2::aes_string(x = "value", fill = "variable"),
                          alpha = 0.4,
                          bw = "bcv") +
    ggplot2::labs(x = "Sens", y = "density(Sens)") +
    ggplot2::xlim(-2 * max(sens$std, na.rm = TRUE), 2 * max(sens$std, na.rm = TRUE))


  # Plot the list of plots created before
  gridExtra::grid.arrange(grobs = plotlist,
                          nrow  = length(plotlist),
                          ncols = 1)
  options(warn = 0)

  if (.returnSens) {
    return(sens)
  }
}

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP train
SensAnalysisMLP.train <- function(MLP.fit, .returnSens = TRUE, ...) {
  actfunc <- c("linear", "sigmoid", ifelse(MLP.fit$modelType == "Regression", "linear", "sigmoid"))
  SensAnalysisMLP.default(MLP.fit$finalModel, trData = MLP.fit$trainingData,
                          actfunc = actfunc,
                          .returnSens = .returnSens, preProc = MLP.fit$preProcess,
                          terms = MLP.fit$terms, ...)
}

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP H2OMultinomialModel
SensAnalysisMLP.H2OMultinomialModel <- function(MLP.fit, .returnSens = TRUE, ...) {
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
  trData <- tryCatch(
    {as.data.frame(eval(parse(text = MLP.fit@parameters$training_frame)))},
    error=function(cond) {
      stop("The training data has not been detected, load the data to the h2o cluster\n")
    },
    finally={}
  )
  # Change the name of the output in trData
  if(MLP.fit@parameters$y %in% names(trData)) {
    names(trData)[which(MLP.fit@parameters$y == names(trData))] <- ".outcome"
  } else {
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
                          preProc = preProc,
                          terms = NULL, ...)
}

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP H2ORegressionModel
SensAnalysisMLP.H2ORegressionModel <- function(MLP.fit, .returnSens = TRUE, ...) {
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
  trData <- tryCatch(
    {as.data.frame(eval(parse(text = MLP.fit@parameters$training_frame)))},
    error=function(cond) {
      stop("The training data has not been detected, load the data to the h2o cluster\n")
    },
    finally={}
  )
  # Change the name of the output in trData
  if(MLP.fit@parameters$y %in% names(trData)) {
    names(trData)[which(MLP.fit@parameters$y == names(trData))] <- ".outcome"
  } else {
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
                          preProc = preProc,
                          terms = NULL, ...)
}

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP NARMAX
SensAnalysisMLP.NARMAX <- function(MLP.fit, .returnSens = TRUE, ...) {
  # for a narmax function
  stop("This function is not done yet")
}


#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP list
SensAnalysisMLP.list <- function(MLP.fit, .returnSens = TRUE, trData,...) {
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
                          preProc = NULL,
                          terms = NULL, ...)
}

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP mlp
SensAnalysisMLP.mlp <- function(MLP.fit, .returnSens = TRUE, trData, ...) {
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
                          preProc = NULL,
                          terms = NULL, ...)
}

#' @rdname SensAnalysisMLP
#'
#' @export
#'
#' @method SensAnalysisMLP nn
SensAnalysisMLP.nn <- function(MLP.fit, .returnSens = TRUE, ...) {
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
                          preProc = NULL,
                          terms = NULL, ...)
}
