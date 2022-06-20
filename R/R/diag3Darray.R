#' Define function to create a 'diagonal' array or get the diagonal of an array
#' @param x \code{number} or \code{vector} defining the value of the diagonal of \code{3D array}
#' @param dim \code{integer} defining the length of the diagonal. Default is \code{length(x)}.
#' If \code{length(x) != 1}, \code{dim} must be equal to \code{length(x)}.
#' @param out \code{character} specifying which type of diagonal to return (\code{"vector"}
#' or \code{"matrix"}). See \code{Details}
#' @return \code{array} with all elements zero except the diagonal, with dimensions c(dim,dim,dim)
#' @details
#' The diagonal of a 3D array has been defined as those elements in positions c(int,int,int),
#' i.e., the three digits are the same.
#'
#' If the diagonal should be returned, \code{out} specifies if it should return a \code{"vector"} with
#' the elements of position c(int,int,int), or \code{"matrix"} with the elements of position c(int,dim,int),
#' i.e., \code{dim = 2} -> elements (1,1,1),(2,1,2),(3,1,3),(1,2,1),(2,2,2),(3,2,3),(3,1,3),(3,2,3),(3,3,3).
#' @examples
#' x <- diag3Darray(c(1,4,6), dim = 3)
#' x
#' # , , 1
#' #
#' # [,1] [,2] [,3]
#' # [1,]    1    0    0
#' # [2,]    0    0    0
#' # [3,]    0    0    0
#' #
#' # , , 2
#' #
#' # [,1] [,2] [,3]
#' # [1,]    0    0    0
#' # [2,]    0    4    0
#' # [3,]    0    0    0
#' #
#' # , , 3
#' #
#' # [,1] [,2] [,3]
#' # [1,]    0    0    0
#' # [2,]    0    0    0
#' # [3,]    0    0    6
#' diag3Darray(x)
#' # 1, 4, 6
#' @export diag3Darray
diag3Darray <- function(x = 1, dim = length(x), out = "vector") {
  if (is.array(x)) {
    if (out == "vector") {
      len.i <- min(dim(x))
      return(x[cbind(1:len.i,1:len.i,1:len.i)])
    } else if (out == "matrix") {
      len.i <- min(dim(x))
      ind.1 <- rep(1:len.i, each = len.i)
      ind.2 <- rep(1:len.i, times = len.i)
      indx <- matrix(ind.1, nrow = len.i^2, ncol = 3)
      indx[,dim] <- ind.2
      return(matrix(x[indx], nrow = len.i, ncol = len.i))
    }
  } else {
    if(length(x) != 1 && dim != length(x)) {
      stop('When x is a vector, dim must be equal to length(x)')
    }
    # Obtain number of elements
    nelem <- dim*dim*dim
    # Obtain positions of diagonal elements
    diag.pos <- seq(1,nelem,length.out = dim)
    # Create elements of array
    elems <- rep(0,nelem)
    elems[diag.pos] <- x
    # Create and return arrays
    return(array(elems, dim = rep(dim,3)))
  }
}
#' Define function to change the diagonal of array
#' @param x  \code{3D array} whose diagonal must be c hanged
#' @param value \code{vector} defining the new values of diagonal.
#' @return \code{array} with all elements zero except the diagonal, with dimensions c(dim,dim,dim)
#' @details The diagonal of a 3D array has been defined as those elements in positions c(int,int,int),
#' i.e., the three digits are the same.
#' @examples
#' x <- array(1, dim = c(3,3,3))
#' diag3Darray(x) <- c(2,2,2)
#' x
#' #  , , 1
#' #
#' #  [,1] [,2] [,3]
#' #  [1,]    2    1    1
#' #  [2,]    1    1    1
#' #  [3,]    1    1    1
#' #
#' #  , , 2
#' #
#' #  [,1] [,2] [,3]
#' #  [1,]    1    1    1
#' #  [2,]    1    2    1
#' #  [3,]    1    1    1
#' #
#' #  , , 3
#' #
#' #  [,1] [,2] [,3]
#' #  [1,]    1    1    1
#' #  [2,]    1    1    1
#' #  [3,]    1    1    2
#' @export `diag3Darray<-`
`diag3Darray<-` <- function(x, value)
{
  dx <- dim(x)
  if (length(dx) != 3L)
    stop("only array diagonals can be replaced")
  len.i <- min(dx)
  len.v <- length(value)
  if (len.v != 1L && len.v != len.i)
    stop("replacement diagonal has wrong length")
  if (len.i) {
    i <- seq_len(len.i)
    x[cbind(i, i, i)] <- value
  }
  x
}
