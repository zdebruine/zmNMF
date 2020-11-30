#' Zero-masked NMF with angular regularization, and (in the future) sparse matrix support
#' 
#' @param A A matrix to be factorized, in dense format (currently, sparse support in the future)
#' @param k Decomposition rank, integer.
#' @param alpha Angular penalty, ideal when factorizing symmetric matrices and W is expected to be nearly equal to H
#' @param rel.tol Stop criterion, defined as the relative tolerance between two successive iterations: |e2-e1|/avg(e1,e2). Default 1e-3.
#' @param inner.rel.tol Stop criterion for the inner sequential coordinate descent least squares loop, defined as relative tolerance passed to inner W or H during matrix updating: |e2-e1|/avg(e1,e2). Default 1e-6.
#' @param max.iter Maximum number of alternating NNLS solutions for H and W, integer. Default 1000.
#' @param inner.max.iter Maximum number of iterations passed to each inner W or H matrix updating function. Default 1e-6.
#' @param verbose Boolean, gives error updates for each trace iterations when TRUE.
#' @param n.threads Number of threads/CPUs to use. Default to 0 (all cores).
#' @param trace An integer indicating how frequently the MSE error should be calculated and checked for convergence. To check error every iteration, specify 1. To avoid checking error at all, specify trace > max.iter.  Default 5.
#' @return A list of W and H matrices
zm_nmf <- function(A, k = NULL, max.iter = 1000, rel.tol = 1e-3, n.threads = 0, verbose = TRUE, inner.max.iter = 100, inner.rel.tol = 1e-6, trace = 5, alpha = 0) {
    if(n.threads < 0) stop("Specify 0 or a positive integer for n.threads")
    if(is.null(k)) stop("Specify a positive integer value for k")
    if(rel.tol > 0.1) warning("rel.tol is greater than 0.1, results may be unstable")
    if(inner.rel.tol > 1e-4) warning("inner.rel.tol is greater than 1e-5, it may take unnecessarily long to converge. Consider a smaller inner.rel.tol.")
    if(trace < 1) stop("trace must be a positive integer")
    if(inner.max.iter < 50) warning("inner.max.iter < 50 is not recommended")
	res <- zm_nmf(A, as.integer(k), as.integer(max.iter), as.double(rel.tol), as.integer(n.threads), as.logical(verbose), as.integer(inner.max.iter), as.double(inner.rel.tol), as.integer(trace), as.double(alpha))
    rownames(res$W) <- rownames(A)
    colnames(res$H) <- colnames(A)
    colnames(res$W) <- rownames(res$H) <- paste0("NMF_",1:k)
    return(res)
}