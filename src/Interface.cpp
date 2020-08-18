#include "Interface.h" 
// [[Rcpp::depends(RcppArmadillo)]]


// [[Rcpp::export]]
Rcpp::List L0LearnFit(const SEXP& X, const arma::vec& y, const std::string Loss, const std::string Penalty,
                      const std::string Algorithm, const unsigned int NnzStopNum, const unsigned int G_ncols,
                      const unsigned int G_nrows, const double Lambda2Max, const double Lambda2Min,
                      const bool PartialSort, const unsigned int MaxIters, const double Tol, const bool ActiveSet,
                      const unsigned int ActiveSetNum, const unsigned int MaxNumSwaps,
                      const double ScaleDownFactor, unsigned int ScreenSize, const bool LambdaU,
                      const std::vector< std::vector<double> > Lambdas,
                      const unsigned int ExcludeFirstK, const bool Intercept) {
  
  
  if (Rf_isS4(X) && Rf_inherits(X, "dgCMatrix")){
    arma::sp_mat m = Rcpp::as<arma::sp_mat>(X);
    return _L0LearnFit(m, y, Loss, Penalty, Algorithm, NnzStopNum, G_ncols, G_nrows, Lambda2Max, Lambda2Min,
                       PartialSort, MaxIters, Tol, ActiveSet, ActiveSetNum, MaxNumSwaps, ScaleDownFactor, ScreenSize, LambdaU,
                       Lambdas, ExcludeFirstK, Intercept);
  } else if (Rf_isArray(X)){
    arma::mat m = Rcpp::as<arma::mat>(X);
    return _L0LearnFit(m, y, Loss, Penalty, Algorithm, NnzStopNum, G_ncols, G_nrows, Lambda2Max, Lambda2Min,
                       PartialSort, MaxIters, Tol, ActiveSet, ActiveSetNum, MaxNumSwaps, ScaleDownFactor, ScreenSize, LambdaU,
                       Lambdas, ExcludeFirstK, Intercept);
  } else {
    to_arma_error();
  }
}
  

// [[Rcpp::export]]
Rcpp::List L0LearnCV(const SEXP& X, const arma::vec& y, const std::string Loss, const std::string Penalty,
                     const std::string Algorithm, const unsigned int NnzStopNum, const unsigned int G_ncols,
                     const unsigned int G_nrows, const double Lambda2Max, const double Lambda2Min,
                     const bool PartialSort, const unsigned int MaxIters, const double Tol, const bool ActiveSet,
                     const unsigned int ActiveSetNum, const unsigned int MaxNumSwaps, const double ScaleDownFactor,
                     unsigned int ScreenSize, const bool LambdaU, const std::vector< std::vector<double> > Lambdas,
                     const unsigned int nfolds, const double seed, const unsigned int ExcludeFirstK,
                     const bool Intercept){
  
  if (Rf_isS4(X) && Rf_inherits(X, "dgCMatrix")){
    arma::sp_mat m = Rcpp::as<arma::sp_mat>(X);
    return _L0LearnCV(m, y, Loss, Penalty,
                      Algorithm, NnzStopNum, G_ncols, G_nrows,
                      Lambda2Max, Lambda2Min, PartialSort,
                      MaxIters, Tol, ActiveSet,
                      ActiveSetNum, MaxNumSwaps,
                      ScaleDownFactor, ScreenSize, LambdaU, Lambdas,
                      nfolds, seed, ExcludeFirstK,Intercept);
    
  } else if (Rf_isArray(X)) {
    arma::mat m = Rcpp::as<arma::mat>(X);
    return _L0LearnCV(m, y, Loss, Penalty,
                      Algorithm, NnzStopNum, G_ncols, G_nrows,
                      Lambda2Max, Lambda2Min, PartialSort,
                      MaxIters, Tol, ActiveSet,
                      ActiveSetNum, MaxNumSwaps,
                      ScaleDownFactor, ScreenSize, LambdaU, Lambdas,
                      nfolds, seed, ExcludeFirstK, Intercept);
  } else {
    to_arma_error();
  }
}
