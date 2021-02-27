#ifndef NORMALIZE_H
#define NORMALIZE_H

#include <tuple>
#include "RcppEigen.h"
#include "utils.h"
#include "BetaVector.h"

std::tuple<beta_vector, double> DeNormalize(beta_vector & B_scaled, 
                                            Eigen::ArrayXd & BetaMultiplier, 
                                            Eigen::ArrayXd & meanX, double meany);

template <typename T>
std::tuple<Eigen::ArrayXd, Eigen::ArrayXd, double, double>  Normalize(const T& X, 
                                                            const Eigen::ArrayXd& y, 
                                                            T& X_normalized, 
                                                            Eigen::ArrayXd & y_normalized, 
                                                            bool Normalizey, 
                                                            bool intercept) {
    
    auto meanX = matrix_center(X, X_normalized, intercept);
    auto scaleX = matrix_normalize(X_normalized);
    
    Eigen::ArrayXd BetaMultiplier;
    double meany = 0;
    double scaley;
    if (Normalizey) {
        if (intercept){
            meany = y.mean();
        }
        y_normalized = y - meany;
        
        // TODO: Use l2_norm
        scaley = y_normalized.matrix().norm(); 
        
        // properly handle cases where y is constant
        if (scaley == 0){
            scaley = 1;
        }
        
        y_normalized = y_normalized / scaley;
        BetaMultiplier = scaley / scaleX.transpose().array(); // transpose scale to get a col vec
        // Multiplying the learned Beta by BetaMultiplier gives the optimal Beta on the original scale
    } else {
        y_normalized = y;
        BetaMultiplier = 1 / scaleX.transpose().array(); // transpose scale to get a col vec
        scaley = 1;
    }

    return std::make_tuple(BetaMultiplier, meanX.transpose(), meany, scaley);
}

#endif // NORMALIZE_H
