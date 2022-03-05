#' Define function to create a 'diagonal' array or get the diagonal of an array
#' @param x \code{number} or \code{vector} defining the value of the diagonal of \code{4D array}
#' @param dim \code{integer} defining the length of the diagonal. Default is \code{length(x)}.
#' If \code{length(x) != 1}, \code{dim} must be equal to \code{length(x)}.
#' @return \code{array} with all elements zero except the diagonal, with dimensions c(dim,dim,dim)
#' @details
#' The diagonal of a 4D array has been defined as those elements in positions c(int,int,int,int),
#' i.e., the four digits are the same.
#'
#' @examples
#' x <- diag4Darray(c(1,3,6,2), dim = 4)
#' x
#' # , , 1, 1
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    1    0    0    0
#' # [2,]    0    0    0    0
#' # [3,]    0    0    0    0
#' # [4,]    0    0    0    0
#' #
#' # , , 2, 1
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    0    0    0    0
#' # [2,]    0    0    0    0
#' # [3,]    0    0    0    0
#' # [4,]    0    0    0    0
#' #
#' # , , 3, 1
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    0    0    0    0
#' # [2,]    0    0    0    0
#' # [3,]    0    0    0    0
#' # [4,]    0    0    0    0
#' #
#' # , , 4, 1
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    0    0    0    0
#' # [2,]    0    0    0    0
#' # [3,]    0    0    0    0
#' # [4,]    0    0    0    0
#' #
#' # , , 1, 2
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    0    0    0    0
#' # [2,]    0    0    0    0
#' # [3,]    0    0    0    0
#' # [4,]    0    0    0    0
#' #
#' # , , 2, 2
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    0    0    0    0
#' # [2,]    0    3    0    0
#' # [3,]    0    0    0    0
#' # [4,]    0    0    0    0
#' #
#' # , , 3, 2
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    0    0    0    0
#' # [2,]    0    0    0    0
#' # [3,]    0    0    0    0
#' # [4,]    0    0    0    0
#' #
#' # , , 4, 2
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    0    0    0    0
#' # [2,]    0    0    0    0
#' # [3,]    0    0    0    0
#' # [4,]    0    0    0    0
#' #
#' # , , 1, 3
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    0    0    0    0
#' # [2,]    0    0    0    0
#' # [3,]    0    0    0    0
#' # [4,]    0    0    0    0
#' #
#' # , , 2, 3
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    0    0    0    0
#' # [2,]    0    0    0    0
#' # [3,]    0    0    0    0
#' # [4,]    0    0    0    0
#' #
#' # , , 3, 3
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    0    0    0    0
#' # [2,]    0    0    0    0
#' # [3,]    0    0    6    0
#' # [4,]    0    0    0    0
#' #
#' # , , 4, 3
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    0    0    0    0
#' # [2,]    0    0    0    0
#' # [3,]    0    0    0    0
#' # [4,]    0    0    0    0
#' #
#' # , , 1, 4
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    0    0    0    0
#' # [2,]    0    0    0    0
#' # [3,]    0    0    0    0
#' # [4,]    0    0    0    0
#' #
#' # , , 2, 4
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    0    0    0    0
#' # [2,]    0    0    0    0
#' # [3,]    0    0    0    0
#' # [4,]    0    0    0    0
#' #
#' # , , 3, 4
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    0    0    0    0
#' # [2,]    0    0    0    0
#' # [3,]    0    0    0    0
#' # [4,]    0    0    0    0
#' #
#' # , , 4, 4
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    0    0    0    0
#' # [2,]    0    0    0    0
#' # [3,]    0    0    0    0
#' # [4,]    0    0    0    2
#' diag4Darray(x)
#' # 1, 3, 6, 2
#' @export diag4Darray
diag4Darray <- function(x = 1, dim = length(x)) {
  if (is.array(x)) {
    len.i <- min(dim(x))
    return(x[cbind(1:len.i,1:len.i,1:len.i,1:len.i)])
  } else {
    if(length(x) != 1 && dim != length(x)) {
      stop('When x is a vector, dim must be equal to length(x)')
    }
    # Obtain number of elements
    nelem <- dim*dim*dim*dim
    # Obtain positions of diagonal elements
    diag.pos <- seq(1,nelem,length.out = dim)
    # Create elements of array
    elems <- rep(0,nelem)
    elems[diag.pos] <- x
    # Create and return arrays
    return(array(elems, dim = rep(dim,4)))
  }
}
#' Define function to change the diagonal of array
#' @param x  \code{3D array} whose diagonal must be c hanged
#' @param value \code{vector} defining the new values of diagonal.
#' @return \code{array} with all elements zero except the diagonal, with dimensions c(dim,dim,dim)
#' @details The diagonal of a 3D array has been defined as those elements in positions c(int,int,int),
#' i.e., the three digits are the same.
#' @examples
#' x <- array(1, dim = c(4,4,4,4))
#' diag4Darray(x) <- c(2,2,2,2)
#' x
#' # , , 1, 1
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    2    1    1    1
#' # [2,]    1    1    1    1
#' # [3,]    1    1    1    1
#' # [4,]    1    1    1    1
#' #
#' # , , 2, 1
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    1    1    1    1
#' # [2,]    1    1    1    1
#' # [3,]    1    1    1    1
#' # [4,]    1    1    1    1
#' #
#' # , , 3, 1
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    1    1    1    1
#' # [2,]    1    1    1    1
#' # [3,]    1    1    1    1
#' # [4,]    1    1    1    1
#' #
#' # , , 4, 1
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    1    1    1    1
#' # [2,]    1    1    1    1
#' # [3,]    1    1    1    1
#' # [4,]    1    1    1    1
#' #
#' # , , 1, 2
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    1    1    1    1
#' # [2,]    1    1    1    1
#' # [3,]    1    1    1    1
#' # [4,]    1    1    1    1
#' #
#' # , , 2, 2
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    1    1    1    1
#' # [2,]    1    2    1    1
#' # [3,]    1    1    1    1
#' # [4,]    1    1    1    1
#' #
#' # , , 3, 2
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    1    1    1    1
#' # [2,]    1    1    1    1
#' # [3,]    1    1    1    1
#' # [4,]    1    1    1    1
#' #
#' # , , 4, 2
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    1    1    1    1
#' # [2,]    1    1    1    1
#' # [3,]    1    1    1    1
#' # [4,]    1    1    1    1
#' #
#' # , , 1, 3
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    1    1    1    1
#' # [2,]    1    1    1    1
#' # [3,]    1    1    1    1
#' # [4,]    1    1    1    1
#' #
#' # , , 2, 3
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    1    1    1    1
#' # [2,]    1    1    1    1
#' # [3,]    1    1    1    1
#' # [4,]    1    1    1    1
#' #
#' # , , 3, 3
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    1    1    1    1
#' # [2,]    1    1    1    1
#' # [3,]    1    1    2    1
#' # [4,]    1    1    1    1
#' #
#' # , , 4, 3
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    1    1    1    1
#' # [2,]    1    1    1    1
#' # [3,]    1    1    1    1
#' # [4,]    1    1    1    1
#' #
#' # , , 1, 4
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    1    1    1    1
#' # [2,]    1    1    1    1
#' # [3,]    1    1    1    1
#' # [4,]    1    1    1    1
#' #
#' # , , 2, 4
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    1    1    1    1
#' # [2,]    1    1    1    1
#' # [3,]    1    1    1    1
#' # [4,]    1    1    1    1
#' #
#' # , , 3, 4
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    1    1    1    1
#' # [2,]    1    1    1    1
#' # [3,]    1    1    1    1
#' # [4,]    1    1    1    1
#' #
#' # , , 4, 4
#' #
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    1    1    1    1
#' # [2,]    1    1    1    1
#' # [3,]    1    1    1    1
#' # [4,]    1    1    1    2
#' @export `diag4Darray<-`
`diag4Darray<-` <- function(x, value)
{
  dx <- dim(x)
  if (length(dx) != 4L)
    stop("only array diagonals can be replaced")
  len.i <- min(dx)
  len.v <- length(value)
  if (len.v != 1L && len.v != len.i)
    stop("replacement diagonal has wrong length")
  if (len.i) {
    i <- seq_len(len.i)
    x[cbind(i, i, i, i)] <- value
  }
  x
}
