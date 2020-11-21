#ifndef NORMALIZE_H
#define NORMALIZE_H

#include <tuple>
#include <RcppEigen.h>
#include "utils.h"

std::tuple<Eigen::SparseMatrix<double>, double> DeNormalize(Eigen::SparseMatrix<double> & B_scaled, 
                                            Eigen::VectorXd & BetaMultiplier, 
                                            Eigen::VectorXd & meanX, double meany);

template <typename T>
std::tuple<T,Eigen::VectorXd,Eigen::VectorXd, double, double>  Normalize(const T& X, 
                                                       constEigen::VectorXd& y, 
                                                      Eigen::VectorXd & y_normalized, 
                                                       bool Normalizey, 
                                                       bool intercept) {
    
    auto martrix_center_return = matrix_center(X, intercept);
    T X_normalized = std::get<0>(martrix_center_return);
    Eigen::VectorXd meanX = std::get<1>(martrix_center_return);
    
    Eigen::VectorXd scaleX = matrix_normalize(X_normalized);
    
   Eigen::VectorXd BetaMultiplier;
    double meany = 0;
    double scaley;
    if (Normalizey) {
        if (intercept){
            meany = arma::mean(y);
        }
        y_normalized = y - meany;
        
        // TODO: Use l2_norm
        scaley = arma::norm(y_normalized, 2);
        
        // properly handle cases where y is constant
        if (scaley == 0){
            scaley = 1;
        }
        
        y_normalized = y_normalized / scaley;
        BetaMultiplier = scaley / (scaleX.t()); // transpose scale to get a col vec
        // Multiplying the learned Beta by BetaMultiplier gives the optimal Beta on the original scale
    } else {
        y_normalized = y;
        BetaMultiplier = 1 / (scaleX.t()); // transpose scale to get a col vec
        scaley = 1;
    }
    return std::make_tuple(X_normalized, BetaMultiplier, meanX.t(), meany, scaley);
}

#endif // NORMALIZE_H
