#include "Normalize.h"

std::tuple<beta_vector, double> DeNormalize(beta_vector & B_scaled, 
                                             Eigen::ArrayXd & BetaMultiplier, 
                                             Eigen::ArrayXd & meanX, double meany) {
    beta_vector B_unscaled = B_scaled.array() * BetaMultiplier;
    double intercept = meany - B_unscaled.cwiseProduct(meanX.matrix()).sum();
    // Matrix Type, Intercept
    // Dense,            True -> meanX = colMeans(X)
    // Dense,            False -> meanX = 0 Vector (meany = 0)
    // Sparse,            True -> meanX = 0 Vector
    // Sparse,            False -> meanX = 0 Vector
    return std::make_tuple(B_unscaled, intercept);
}
