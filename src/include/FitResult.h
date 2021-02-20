#ifndef FITRESULT_H
#define FITRESULT_H
#include "RcppEigen.h"
#include "BetaVector.h"

template <class T> // Forward Reference to prevent circular dependencies
class CDBase;

template <typename T>
struct FitResult {
    double Objective;
    beta_vector B;
    CDBase<T> * Model;
    std::size_t IterNum;
    Eigen::ArrayXd r;
    std::vector<double> ModelParams;
    double b0 = 0; // used by classification models and sparse regression models
    Eigen::ArrayXd ExpyXB; // Used by Logistic regression
    Eigen::ArrayXd onemyxb; // Used by SquaredHinge regression
};

#endif
