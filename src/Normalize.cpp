#include "Normalize.h"

std::tuple<Eigen::SparseMatrix<double>, double> DeNormalize(Eigen::SparseMatrix<double> & B_scaled, 
                                            Eigen::VectorXd & BetaMultiplier, 
                                            Eigen::VectorXd & meanX, double meany) {
    Eigen::SparseMatrix<double> B_unscaled = B_scaled % BetaMultiplier;
    double intercept = meany - arma::dot(B_unscaled, meanX);
    // Matrix Type, Intercept
    // Dense,            True -> meanX = colMeans(X)
    // Dense,            False -> meanX = 0 Vector (meany = 0)
    // Sparse,            True -> meanX = 0 Vector
    // Sparse,            False -> meanX = 0 Vector
    return std::make_tuple(B_unscaled, intercept);
}
