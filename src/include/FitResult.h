#ifndef FITRESULT_H
#define FITRESULT_H
#include <RcppEigen.h>

template <class T> // Forward Reference to prevent circular dependencies
class CD;

template <typename T>
struct FitResult {
    double Objective;
    Eigen::SparseMatrix<double> B;
    CD<T> * Model;
    std::size_t IterNum;
   Eigen::VectorXd * r;
    std::vector<double> ModelParams;
    double b0 = 0; // used by classification models and sparse regression models
   Eigen::VectorXd ExpyXB; // Used by Logistic regression
   Eigen::VectorXd onemyxb; // Used by SquaredHinge regression
};

#endif
