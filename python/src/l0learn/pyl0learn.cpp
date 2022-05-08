#include "pyl0learn.h"

namespace py = pybind11;

py_fitmodel L0LearnFit_sparse_wrapper(const sparse_mat &X,
                                    const arma::vec &y,
                                    const std::string & Loss,
                                    const std::string & Penalty,
                                    const std::string & Algorithm,
                                    const unsigned int NnzStopNum,
                                    const unsigned int G_ncols,
                                    const unsigned int G_nrows,
                                    const double Lambda2Max,
                                    const double Lambda2Min,
                                    const bool PartialSort,
                                    const unsigned int MaxIters,
                                    const double rtol,
                                    const double atol,
                                    const bool ActiveSet,
                                    const unsigned int ActiveSetNum,
                                    const unsigned int MaxNumSwaps,
                                    const double ScaleDownFactor,
                                    const unsigned int ScreenSize,
                                    const bool LambdaU,
                                    const std::vector<std::vector<double>> & Lambdas,
                                    const unsigned int ExcludeFirstK,
                                    const bool Intercept,
                                    const bool withBounds,
                                    const arma::vec &Lows,
                                    const arma::vec &Highs) {
    return py_fitmodel(L0LearnFit<arma::sp_mat>(X.to_arma_object(),
                                   y,
                                   Loss,
                                   Penalty,
                                   Algorithm,
                                   NnzStopNum,
                                   G_ncols,
                                   G_nrows,
                                   Lambda2Max,
                                   Lambda2Min,
                                   PartialSort,
                                   MaxIters,
                                   rtol,
                                   atol,
                                   ActiveSet,
                                   ActiveSetNum,
                                   MaxNumSwaps,
                                   ScaleDownFactor,
                                   ScreenSize,
                                   LambdaU,
                                   Lambdas,
                                   ExcludeFirstK,
                                   Intercept,
                                   withBounds,
                                   Lows,
                                   Highs));
}

py_fitmodel L0LearnFit_dense_wrapper(const arma::mat &X,
                                      const arma::vec &y,
                                      const std::string & Loss,
                                      const std::string & Penalty,
                                      const std::string & Algorithm,
                                      const unsigned int NnzStopNum,
                                      const unsigned int G_ncols,
                                      const unsigned int G_nrows,
                                      const double Lambda2Max,
                                      const double Lambda2Min,
                                      const bool PartialSort,
                                      const unsigned int MaxIters,
                                      const double rtol,
                                      const double atol,
                                      const bool ActiveSet,
                                      const unsigned int ActiveSetNum,
                                      const unsigned int MaxNumSwaps,
                                      const double ScaleDownFactor,
                                      const unsigned int ScreenSize,
                                      const bool LambdaU,
                                      const std::vector<std::vector<double>> & Lambdas,
                                      const unsigned int ExcludeFirstK,
                                      const bool Intercept,
                                      const bool withBounds,
                                      const arma::vec &Lows,
                                      const arma::vec &Highs) {
  return py_fitmodel(L0LearnFit<arma::mat>(X,
                                   y,
                                   Loss,
                                   Penalty,
                                   Algorithm,
                                   NnzStopNum,
                                   G_ncols,
                                   G_nrows,
                                   Lambda2Max,
                                   Lambda2Min,
                                   PartialSort,
                                   MaxIters,
                                   rtol,
                                   atol,
                                   ActiveSet,
                                   ActiveSetNum,
                                   MaxNumSwaps,
                                   ScaleDownFactor,
                                   ScreenSize,
                                   LambdaU,
                                   Lambdas,
                                   ExcludeFirstK,
                                   Intercept,
                                   withBounds,
                                   Lows,
                                   Highs));
}

py_cvfitmodel L0LearnCV_dense_wrapper(const arma::mat &X,
                                       const arma::vec &y,
                                       const std::string & Loss,
                                       const std::string & Penalty,
                                       const std::string & Algorithm,
                                       const unsigned int NnzStopNum,
                                       const unsigned int G_ncols,
                                       const unsigned int G_nrows,
                                       const double Lambda2Max,
                                       const double Lambda2Min,
                                       const bool PartialSort,
                                       const unsigned int MaxIters,
                                       const double rtol,
                                       const double atol,
                                       const bool ActiveSet,
                                       const unsigned int ActiveSetNum,
                                       const unsigned int MaxNumSwaps,
                                       const double ScaleDownFactor,
                                       const unsigned int ScreenSize,
                                       const bool LambdaU,
                                       const std::vector<std::vector<double>> & Lambdas,
                                       const unsigned int nfolds,
                                       const size_t seed,
                                       const unsigned int ExcludeFirstK,
                                       const bool Intercept,
                                       const bool withBounds,
                                       const arma::vec &Lows,
                                       const arma::vec &Highs) {
  return py_cvfitmodel(L0LearnCV<arma::mat>(X,
                                  y,
                                  Loss,
                                  Penalty,
                                  Algorithm,
                                  NnzStopNum,
                                  G_ncols,
                                  G_nrows,
                                  Lambda2Max,
                                  Lambda2Min,
                                  PartialSort,
                                  MaxIters,
                                  rtol,
                                  atol,
                                  ActiveSet,
                                  ActiveSetNum,
                                  MaxNumSwaps,
                                  ScaleDownFactor,
                                  ScreenSize,
                                  LambdaU,
                                  Lambdas,
                                  nfolds,
                                  seed,
                                  ExcludeFirstK,
                                  Intercept,
                                  withBounds,
                                  Lows,
                                  Highs));
}

py_cvfitmodel L0LearnCV_sparse_wrapper(const sparse_mat &X,
                                       const arma::vec &y,
                                       const std::string & Loss,
                                       const std::string & Penalty,
                                    const std::string & Algorithm,
                                    const unsigned int NnzStopNum,
                                    const unsigned int G_ncols,
                                    const unsigned int G_nrows,
                                    const double Lambda2Max,
                                    const double Lambda2Min,
                                    const bool PartialSort,
                                    const unsigned int MaxIters,
                                    const double rtol,
                                    const double atol,
                                    const bool ActiveSet,
                                    const unsigned int ActiveSetNum,
                                    const unsigned int MaxNumSwaps,
                                    const double ScaleDownFactor,
                                    const unsigned int ScreenSize,
                                    const bool LambdaU,
                                    const std::vector<std::vector<double>> & Lambdas,
                                    const unsigned int nfolds,
                                    const size_t seed,
                                    const unsigned int ExcludeFirstK,
                                    const bool Intercept,
                                    const bool withBounds,
                                    const arma::vec &Lows,
                                    const arma::vec &Highs) {
  return py_cvfitmodel(L0LearnCV<arma::sp_mat>(X.to_arma_object(),
                                 y,
                                 Loss,
                                 Penalty,
                                 Algorithm,
                                 NnzStopNum,
                                 G_ncols,
                                 G_nrows,
                                 Lambda2Max,
                                 Lambda2Min,
                                 PartialSort,
                                 MaxIters,
                                 rtol,
                                 atol,
                                 ActiveSet,
                                 ActiveSetNum,
                                 MaxNumSwaps,
                                 ScaleDownFactor,
                                 ScreenSize,
                                 LambdaU,
                                 Lambdas,
                                 nfolds,
                                 seed,
                                 ExcludeFirstK,
                                 Intercept,
                                 withBounds,
                                 Lows,
                                 Highs));
}


PYBIND11_MODULE(l0learn_core, m) {
  m.attr("__name__") = "l0learn.l0learn_core";

  m.def("_L0LearnFit_dense", &L0LearnFit_dense_wrapper);

  m.def("_L0LearnFit_sparse", &L0LearnFit_sparse_wrapper);

  m.def("_L0LearnCV_dense", &L0LearnCV_dense_wrapper);

  m.def("_L0LearnCV_sparse", &L0LearnCV_sparse_wrapper);

  py::class_<sparse_mat>(m, "_sparse_mat")
      .def(py::init<const arma::uvec &,
           const arma::uvec &,
           const arma::vec &,
           const arma::uword,
           const arma::uword>());

  py::class_<py_fitmodel>(m, "_py_fitmodel")
      .def(py::init<fitmodel const &>())
      .def_readonly("Lambda0", &py_fitmodel::Lambda0)
      .def_readonly("Lambda12", &py_fitmodel::Lambda12)
      .def_readonly("NnzCount", &py_fitmodel::NnzCount)
      .def_readonly("Beta", &py_fitmodel::Beta)
      .def_readonly("Intercept", &py_fitmodel::Intercept)
      .def_readonly("Converged", &py_fitmodel::Converged);

  py::class_<py_cvfitmodel>(m, "_py_cvfitmodel")
    .def(py::init<cvfitmodel const &>())
    .def_readonly("Lambda0", &py_cvfitmodel::Lambda0)
    .def_readonly("Lambda12", &py_cvfitmodel::Lambda12)
    .def_readonly("NnzCount", &py_cvfitmodel::NnzCount)
    .def_readonly("Beta", &py_cvfitmodel::Beta)
    .def_readonly("Intercept", &py_cvfitmodel::Intercept)
    .def_readonly("Converged", &py_cvfitmodel::Converged)
    .def_readonly("CVMeans", &py_cvfitmodel::CVMeans)
    .def_readonly("CVSDs", &py_cvfitmodel::CVSDs);
}
