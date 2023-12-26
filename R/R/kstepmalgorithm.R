#' Find Critical Value
#'
#' This function finds the smallest x such that the probability of a random variable
#' being less than or equal to x is greater than or equal to 1 - alpha.
#' It uses the uniroot function to find where the empirical cumulative distribution function (ECDF)
#' crosses 1 - alpha.
#'
#' @param ecdf_func An ECDF function representing the distribution of a random variable.
#' @param alpha A numeric value specifying the significance level.
#'
#' @return The smallest x such that P(X <= x) >= 1 - alpha.
#' @examples
#' data <- rnorm(100)
#' ecdf_data <- ecdf(data)
#' critical_val <- find_critical_value(ecdf_data, 0.05)
#' @export
find_critical_value <- function(ecdf_func, alpha) {
    # This function will find the smallest x such that P(X <= x) >= 1 - alpha
    # We use the uniroot function to find where the ECDF crosses 1 - alpha
    lower_bound <- stats::quantile(ecdf_func, 0)  # The minimum of the kmax_nK_b values
    upper_bound <- stats::quantile(ecdf_func, 1)  # The maximum of the kmax_nK_b values

    # Use uniroot to find where the function crosses 1 - alpha
    critical_value <- stats::uniroot(function(x) ecdf_func(x) - (1 - alpha),
                              interval = c(lower_bound, upper_bound),
                              tol = .Machine$double.eps^0.5)$root
    return(critical_value)
}

#' k-StepM Algorithm for Hypothesis Testing
#'
#' This function implements the k-stepM algorithm for multiple hypothesis testing.
#' It tests each hypothesis using the critical value calculated from the ECDF
#' of the k-max differences, updating the critical value, and iterating until all hypotheses
#' are tested.
#'
#' @param original_stats A numeric vector of original test statistics for each hypothesis.
#' @param bootstrap_stats A numeric matrix of bootstrap test statistics, with rows representing
#'        bootstrap samples and columns representing hypotheses.
#' @param num_hypotheses An integer specifying the total number of hypotheses.
#' @param alpha A numeric value specifying the significance level.
#' @param k An integer specifying the threshold number for controlling the k-familywise error rate.
#'
#' @return A list containing two elements: 'signif', a logical vector indicating which hypotheses
#'         are rejected, and 'cv', a numeric vector of critical values used for each hypothesis.
#' @examples
#' original_stats <- rnorm(10)
#' bootstrap_stats <- matrix(rnorm(1000), ncol = 10)
#' result <- kStepMAlgorithm(original_stats, bootstrap_stats, 10, 0.05, 1)
#' @export
kStepMAlgorithm <- function(original_stats, bootstrap_stats, num_hypotheses, alpha, k) {
    # Assuming Tnj is a vector of test statistics for each hypothesis
    ordered_indices <- order(original_stats, decreasing = TRUE)
    ordered_Tnj <- original_stats[ordered_indices, drop=FALSE]
    ordered_Tnj_b <- bootstrap_stats[,ordered_indices, drop=FALSE]

    # Store critical values
    c_kv <- numeric(num_hypotheses)

    # Configuration
    B <- nrow(bootstrap_stats)

    # for (inp in 1:num_hypotheses) {
    while(sum(c_kv == 0) >= 1) {
        # Initialize a vector to store the kmax for each bootstrap sample
        kmax_nK_b <- matrix(nrow=B, ncol=1)

        K <- c_kv[ordered_indices] == 0
        for (b in 1:B) {
            # Calculate the differences for this bootstrap sample
            differences <- ordered_Tnj_b[b, K] - ordered_Tnj[K]

            # Find the k-th largest difference within the subset K
            sorted_differences <- sort(differences, decreasing = TRUE)
            kmax_nK_b[b, 1] <- sorted_differences[k]
        }

        # Step 1: Create the ECDF for the k-th largest values
        if (length(unique(kmax_nK_b)) > 2) {
            ecdf_kmax <- stats::ecdf(kmax_nK_b)
            c_K <- abs(find_critical_value(ecdf_kmax, alpha=alpha))
        } else {
            c_K <- abs(max(kmax_nK_b))
        }

        R1 <- sum((original_stats[c_kv == 0] - c_K) > 0) # Number of rejected hypothesis
        if (R1 < k) {
            break
        }

        c_kv[(c_kv == 0) & ((original_stats - c_K) > 0)] = c_K
    }
    c_kv[(c_kv == 0)] <- c_K

    diff <- matrix(nrow=dim(bootstrap_stats)[1], ncol=dim(bootstrap_stats)[2])
    for (b in 1:nrow(bootstrap_stats)) {
        diff[b,] <- bootstrap_stats[b,] - original_stats
    }

    return(list(signif=(original_stats - c_K) > 0, cv = c_kv, alpha=alpha))
}
