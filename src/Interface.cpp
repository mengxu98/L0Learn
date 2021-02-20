#include "Interface.h"
// [[Rcpp::depends(RcppEigen)]]

#include <chrono>
#include <thread>

template <typename T>
GridParams<T> makeGridParams(const std::string Loss, const std::string Penalty,
                             const std::string Algorithm, const std::size_t NnzStopNum, const std::size_t G_ncols,
                             const std::size_t G_nrows, const double Lambda2Max, const double Lambda2Min,
                             const bool PartialSort, const std::size_t MaxIters, const double rtol,
                             const double atol, const bool ActiveSet, const std::size_t ActiveSetNum,
                             const std::size_t MaxNumSwaps, const double ScaleDownFactor,
                             const std::size_t ScreenSize, const bool LambdaU,
                             const std::vector< std::vector<double> > Lambdas,
                             const std::size_t ExcludeFirstK, const bool Intercept,
                             const bool withBounds, const Eigen::ArrayXd &Lows, const Eigen::ArrayXd &Highs){
  GridParams<T> PG;
  PG.NnzStopNum = NnzStopNum;
  PG.G_ncols = G_ncols;
  PG.G_nrows = G_nrows;
  PG.Lambda2Max = Lambda2Max;
  PG.Lambda2Min = Lambda2Min;
  PG.LambdaMinFactor = Lambda2Min; //
  PG.PartialSort = PartialSort;
  PG.ScaleDownFactor = ScaleDownFactor;
  PG.LambdaU = LambdaU;
  PG.LambdasGrid = Lambdas;
  PG.Lambdas = Eigen::VectorXd::Map(Lambdas[0].data(), Lambdas[0].size()); // to handle the case of L0 (i.e., Grid1D)
  PG.intercept = Intercept;

  Params<T> P;
  PG.P = P;
  PG.P.MaxIters = MaxIters;
  PG.P.rtol = rtol;
  PG.P.atol = atol;
  PG.P.ActiveSet = ActiveSet;
  PG.P.ActiveSetNum = ActiveSetNum;
  PG.P.MaxNumSwaps = MaxNumSwaps;
  PG.P.ScreenSize = ScreenSize;
  PG.P.NoSelectK = ExcludeFirstK;
  PG.P.intercept = Intercept;
  PG.P.withBounds = withBounds;
  PG.P.Lows = Lows;
  PG.P.Highs = Highs;

  if (Loss == "SquaredError") {
    PG.P.Specs.SquaredError = true;
  } else if (Loss == "Logistic") {
    PG.P.Specs.Logistic = true;
    PG.P.Specs.Classification = true;
  } else if (Loss == "SquaredHinge") {
    PG.P.Specs.SquaredHinge = true;
    PG.P.Specs.Classification = true;
  }

  if (Algorithm == "CD") {
    PG.P.Specs.CD = true;
  } else if (Algorithm == "CDPSI") {
    PG.P.Specs.PSI = true;
  }

  if (Penalty == "L0") {
    PG.P.Specs.L0 = true;
  } else if (Penalty == "L0L2") {
    PG.P.Specs.L0L2 = true;
  } else if (Penalty == "L0L1") {
    PG.P.Specs.L0L1 = true;
  }
  return PG;
}


template <typename T>
Rcpp::List _L0LearnFit(const T& X, const Eigen::ArrayXd& y, const std::string Loss, const std::string Penalty,
                       const std::string Algorithm, const std::size_t NnzStopNum, const std::size_t G_ncols,
                       const std::size_t G_nrows, const double Lambda2Max, const double Lambda2Min,
                       const bool PartialSort, const std::size_t MaxIters, const double rtol, const double atol,
                       const bool ActiveSet, const std::size_t ActiveSetNum, const std::size_t MaxNumSwaps,
                       const double ScaleDownFactor, const std::size_t ScreenSize, const bool LambdaU,
                       const std::vector< std::vector<double> > Lambdas, const std::size_t ExcludeFirstK,
                       const bool Intercept,  const bool withBounds, const Eigen::ArrayXd &Lows,
                       const Eigen::ArrayXd &Highs){

  int i = 0;
  Rcpp::Rcout << ++i << "\n";
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  GridParams<T> PG = makeGridParams<T>(Loss, Penalty, Algorithm, NnzStopNum, G_ncols, G_nrows,
                      Lambda2Max, Lambda2Min, PartialSort, MaxIters, rtol, atol, ActiveSet,
                      ActiveSetNum, MaxNumSwaps, ScaleDownFactor, ScreenSize,
                      LambdaU, Lambdas, ExcludeFirstK, Intercept, withBounds, Lows, Highs);

  Rcpp::Rcout << ++i << "\n";
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  Grid<T> G(X, y, PG);
  Rcpp::Rcout << ++i << "\n";
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  G.Fit();

  std::string FirstParameter = "lambda";
  std::string SecondParameter = "gamma";

  // Next Construct the list of Sparse Beta Matrices.

  Rcpp::Rcout << ++i << "\n";
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  auto p = X.cols();
  std::vector<Eigen::SparseMatrix<double>> Bs(G.Lambda12.size());

  for (std::size_t i=0; i<G.Lambda12.size(); ++i) {
    // create the px(reg path size) sparse sparseMatrix
    Eigen::SparseMatrix<double> B(p,G.Solutions[i].size());
    for (unsigned int j=0; j<G.Solutions[i].size(); ++j) {
      B.col(j) = G.Solutions[i][j];
    }

    // append the sparse matrix
    Bs.push_back(B);
  }
  Rcpp::Rcout << ++i << "\n";
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  Rcpp::List l = Rcpp::List::create(Rcpp::Named(FirstParameter) = G.Lambda0,
                                    Rcpp::Named(SecondParameter) = G.Lambda12, // contains 0 in case of L0
                                    Rcpp::Named("SuppSize") = G.NnzCount,
                                    Rcpp::Named("beta") = Bs,
                                    Rcpp::Named("a0") = G.Intercepts,
                                    Rcpp::Named("Converged") = G.Converged);

  return l;

}

template <typename T>
Rcpp::List _L0LearnCV(const T& X, const Eigen::ArrayXd& y, const std::string Loss,
                      const std::string Penalty, const std::string Algorithm,
                      const unsigned int NnzStopNum, const unsigned int G_ncols,
                      const unsigned int G_nrows, const double Lambda2Max,
                      const double Lambda2Min, const bool PartialSort,
                      const unsigned int MaxIters, const double rtol,
                      const double atol, const bool ActiveSet,
                      const unsigned int ActiveSetNum,
                      const unsigned int MaxNumSwaps, const double ScaleDownFactor,
                      const unsigned int ScreenSize, const bool LambdaU,
                      const std::vector< std::vector<double> > Lambdas,
                      const unsigned int nfolds, const double seed,
                      const unsigned int ExcludeFirstK, const bool Intercept,
                      const bool withBounds, const Eigen::ArrayXd &Lows,
                      const Eigen::ArrayXd &Highs){

  GridParams<T> PG = makeGridParams<T>(Loss, Penalty, Algorithm, NnzStopNum, G_ncols, G_nrows,
                                       Lambda2Max, Lambda2Min, PartialSort, MaxIters, rtol, atol,
                                       ActiveSet,ActiveSetNum, MaxNumSwaps, ScaleDownFactor,
                                       ScreenSize,LambdaU, Lambdas, ExcludeFirstK, Intercept,
                                       withBounds, Lows, Highs);

  Grid<T> G(X, y, PG);
  G.Fit();

  std::string FirstParameter = "lambda";
  std::string SecondParameter = "gamma";

  // Next Construct the list of Sparse Beta Matrices.

  auto p = X.cols();
  auto n = X.rows();
  std::vector<Eigen::SparseMatrix<double>> Bs(G.Lambda12.size());

  for (std::size_t i=0; i<G.Lambda12.size(); ++i) {
    // create the px(reg path size) sparse sparseMatrix
    Eigen::SparseMatrix<double> B(p, G.Solutions[i].size());

    std::vector<std::size_t> reserved_size(G.Solutions[i].size());
    for (std::size_t j=0; j<G.Solutions[i].size(); ++j)
      reserved_size.push_back(G.Solutions[i][j].nonZeros());

    B.reserve(reserved_size);
    for (std::size_t j=0; j<G.Solutions[i].size(); ++j)
      B.col(j) = G.Solutions[i][j];

    // append the sparse matrix
    Bs[i] = B;
  }


  // CV starts here

  // arma::arma_rng::set_seed(seed);

  //Solutions = std::vector< std::vector<arma::sp_mat> >(G.size());
  //Intercepts = std::vector< std::vector<double> >(G.size());

  std::size_t Ngamma = G.Lambda12.size();

  std::vector<Eigen::MatrixXd> CVError (G.Solutions.size());
  //arma::field< arma::mat > CVError (G.Solutions.size());

  for (std::size_t i=0; i<G.Solutions.size(); ++i)
    CVError.push_back(Eigen::MatrixXd::Zero(G.Lambda0[i].size(), nfolds));

  std::vector<std::size_t> indicies(X.rows());
  std::iota(indicies.begin(), indicies.end(), 0);
  std::random_shuffle(indicies.begin(), indicies.end());

  std::size_t samplesperfold = std::ceil(n/double(nfolds));
  std::size_t samplesinlastfold = samplesperfold - (samplesperfold*nfolds - n);

  std::vector<std::size_t> fullindices(X.rows());
  std::iota(fullindices.begin(), fullindices.end(), 0);


  for (std::size_t j=0; j<nfolds;++j) {
    std::vector<std::size_t> validationindices;
    if (j < nfolds-1)
      validationindices.resize(samplesperfold);
    else
      validationindices.resize(samplesinlastfold);

    std::iota(validationindices.begin(), validationindices.end(), samplesperfold*j);

    std::vector<std::size_t> trainingindices;

    std::set_difference(fullindices.begin(),fullindices.end(),
                        validationindices.begin(), validationindices.end(),
                        std::inserter(trainingindices, trainingindices.begin()));


    // validation_indices contains the randomly permuted validation indices
    std::vector<std::size_t> validation_indices = vector_subset(indicies, validationindices);

    // training_indices is similar to validation_indices but for training
    std::vector<std::size_t> training_indices = vector_subset(indicies, trainingindices);
    
    T Xtraining = matrix_rows_get(X, training_indices);
    Eigen::ArrayXd ytraining = matrix_rows_get(y, training_indices);

    T Xvalidation = matrix_rows_get(X, validation_indices);
    Eigen::ArrayXd yvalidation = matrix_rows_get(y, validation_indices);


    PG.LambdaU = true;
    PG.XtrAvailable = false; // reset XtrAvailable since its changed upon every call
    PG.LambdasGrid = G.Lambda0;
    PG.NnzStopNum = p+1; // remove any constraints on the supp size when fitting over the cv folds // +1 is imp to avoid =p edge case
    if (PG.P.Specs.L0 == true){
      PG.Lambdas = Eigen::VectorXd::Map(PG.LambdasGrid[0].data(), PG.LambdasGrid[0].size());
    }
    Grid<T> Gtraining(Xtraining, ytraining, PG);
    Gtraining.Fit();

    for (std::size_t i=0; i<Ngamma; ++i) {
      // i indexes the gamma parameter
      for (std::size_t k=0; k<Gtraining.Lambda0[i].size(); ++k){
        // k indexes the solutions for a specific gamma

        const Eigen::SparseVector<double> B = Gtraining.Solutions[i][k];
        const double b0 = Gtraining.Intercepts[i][k];
        const Eigen::ArrayXd preds = make_predicitions(X, B, b0);
      
        if (PG.P.Specs.SquaredError) {

          Eigen::ArrayXd r = yvalidation - preds;
          CVError[i](k,j) = r.square().sum() / yvalidation.rows();

        } else if (PG.P.Specs.Logistic) {
          Eigen::ArrayXd ExpyXB = (yvalidation * preds).exp();
          CVError[i](k,j) = (1 + 1 / ExpyXB).log().sum() / yvalidation.rows();
          //std::cout<<"i, j, k"<<i<<" "<<j<<" "<<k<<" CVError[i](k,j): "<<CVError[i](k,j)<<std::endl;
          //CVError[i].print();
        } else if (PG.P.Specs.SquaredHinge) {
          Eigen::ArrayXd onemyxb = 1 - yvalidation * preds;
          CVError[i](k,j) = onemyxb.max(0).square().sum() / yvalidation.rows();
        }
      }
    }
  }

  std::vector<Eigen::ArrayXd> CVMeans(Ngamma);
  std::vector<Eigen::ArrayXd> CVSDs(Ngamma);

  for (std::size_t i=0; i<Ngamma; ++i) {
    //CVMeans[i] = arma::mean(CVError[i],1);
    CVMeans[i] = CVError[i].rowwise().mean();
    //CVSDs[i] = arma::stddev(CVError[i],0,1);
    CVSDs[i] = ((CVError[i].rowwise() - CVError[i].colwise().mean()).array().square().colwise().sum()/(CVError[i].cols()-1)).sqrt();
    //CVSDs[i] = ((CVError[i].array().rowwise() - CVError[i].rowwise().mean()).array().square().sum()/(CVError[i].cols()-1));
  }

  return Rcpp::List::create(Rcpp::Named(FirstParameter) = G.Lambda0,
                            Rcpp::Named(SecondParameter) = G.Lambda12, // contains 0 in case of L0
                            Rcpp::Named("SuppSize") = G.NnzCount,
                            Rcpp::Named("beta") = Bs,
                            Rcpp::Named("a0") = G.Intercepts,
                            Rcpp::Named("Converged") = G.Converged,
                            Rcpp::Named("CVMeans") = CVMeans,
                            Rcpp::Named("CVSDs") = CVSDs );
}


// [[Rcpp::export]]
Rcpp::List L0LearnFit_sparse(const Eigen::SparseMatrix<double>& X,
                             const Eigen::ArrayXd& y, const std::string Loss,
                             const std::string Penalty, const std::string Algorithm,
                             const std::size_t NnzStopNum, const std::size_t G_ncols,
                             const std::size_t G_nrows, const double Lambda2Max,
                             const double Lambda2Min, const bool PartialSort,
                             const std::size_t MaxIters, const double rtol,
                             const double atol, const bool ActiveSet,
                             const std::size_t ActiveSetNum,
                             const std::size_t MaxNumSwaps,
                             const double ScaleDownFactor,
                             const std::size_t ScreenSize, const bool LambdaU,
                             const std::vector< std::vector<double> > Lambdas,
                             const std::size_t ExcludeFirstK, const bool Intercept,
                             const bool withBounds, const Eigen::ArrayXd &Lows,
                             const Eigen::ArrayXd &Highs) {

  return _L0LearnFit(X, y, Loss, Penalty, Algorithm, NnzStopNum, G_ncols, G_nrows, Lambda2Max, Lambda2Min,
                     PartialSort, MaxIters, rtol, atol, ActiveSet, ActiveSetNum, MaxNumSwaps, ScaleDownFactor, ScreenSize, LambdaU,
                     Lambdas, ExcludeFirstK, Intercept, withBounds, Lows, Highs);
}


// [[Rcpp::export]]
Rcpp::List L0LearnFit_dense(const Eigen::MatrixXd& X, const Eigen::ArrayXd& y,
                            const std::string Loss, const std::string Penalty,
                            const std::string Algorithm, const std::size_t NnzStopNum,
                            const std::size_t G_ncols, const std::size_t G_nrows,
                            const double Lambda2Max, const double Lambda2Min,
                            const bool PartialSort, const std::size_t MaxIters,
                            const double rtol, const double atol, const bool ActiveSet,
                            const std::size_t ActiveSetNum, const std::size_t MaxNumSwaps,
                            const double ScaleDownFactor, const std::size_t ScreenSize,
                            const bool LambdaU, const std::vector< std::vector<double> > Lambdas,
                            const std::size_t ExcludeFirstK, const bool Intercept,
                            const bool withBounds, const Eigen::ArrayXd &Lows,
                            const Eigen::ArrayXd &Highs) {

      return _L0LearnFit(X, y, Loss, Penalty, Algorithm, NnzStopNum, G_ncols, G_nrows, Lambda2Max, Lambda2Min,
                       PartialSort, MaxIters, rtol, atol, ActiveSet, ActiveSetNum, MaxNumSwaps, ScaleDownFactor, ScreenSize, LambdaU,
                       Lambdas, ExcludeFirstK, Intercept, withBounds, Lows, Highs);
}


// [[Rcpp::export]]
Rcpp::List L0LearnCV_sparse(const Eigen::SparseMatrix<double>& X, const Eigen::ArrayXd& y,
                            const std::string Loss, const std::string Penalty,
                            const std::string Algorithm, const std::size_t NnzStopNum,
                            const std::size_t G_ncols, const std::size_t G_nrows,
                            const double Lambda2Max, const double Lambda2Min,
                            const bool PartialSort, const std::size_t MaxIters,
                            const double rtol, const double atol, const bool ActiveSet,
                            const std::size_t ActiveSetNum, const std::size_t MaxNumSwaps,
                            const double ScaleDownFactor, const std::size_t ScreenSize,
                            const bool LambdaU, const std::vector< std::vector<double> > Lambdas,
                            const std::size_t nfolds, const double seed,
                            const std::size_t ExcludeFirstK, const bool Intercept,
                            const bool withBounds, const Eigen::ArrayXd &Lows,
                            const Eigen::ArrayXd &Highs){

    return _L0LearnCV(X, y, Loss, Penalty,
                      Algorithm, NnzStopNum, G_ncols, G_nrows,
                      Lambda2Max, Lambda2Min, PartialSort,
                      MaxIters, rtol,atol, ActiveSet,
                      ActiveSetNum, MaxNumSwaps,
                      ScaleDownFactor, ScreenSize, LambdaU, Lambdas,
                      nfolds, seed, ExcludeFirstK, Intercept, withBounds, Lows, Highs);
}

// [[Rcpp::export]]
Rcpp::List L0LearnCV_dense(const Eigen::MatrixXd& X, const Eigen::ArrayXd& y,
                           const std::string Loss, const std::string Penalty,
                           const std::string Algorithm, const std::size_t NnzStopNum,
                           const std::size_t G_ncols, const std::size_t G_nrows,
                           const double Lambda2Max, const double Lambda2Min,
                           const bool PartialSort, const std::size_t MaxIters,
                           const double rtol, const double atol, const bool ActiveSet,
                           const std::size_t ActiveSetNum, const std::size_t MaxNumSwaps,
                           const double ScaleDownFactor, const std::size_t ScreenSize,
                           const bool LambdaU, const std::vector< std::vector<double> > Lambdas,
                           const std::size_t nfolds, const double seed,
                           const std::size_t ExcludeFirstK, const bool Intercept,
                           const bool withBounds, const Eigen::ArrayXd &Lows,
                           const Eigen::ArrayXd &Highs){

  return _L0LearnCV(X, y, Loss, Penalty,
                    Algorithm, NnzStopNum, G_ncols, G_nrows,
                    Lambda2Max, Lambda2Min, PartialSort,
                    MaxIters, rtol,atol, ActiveSet,
                    ActiveSetNum, MaxNumSwaps,
                    ScaleDownFactor, ScreenSize, LambdaU, Lambdas,
                    nfolds, seed, ExcludeFirstK, Intercept, withBounds, Lows, Highs);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cor_matrix(const int p, const double base_cor) {
  Rcpp::NumericMatrix cor(p, p);
  for (int i = 0; i < p; i++){
    for (int j = 0; j < p; j++){
      cor(i, j) = std::pow(base_cor, std::abs(i - j));
    }
  }
  return cor;
}

