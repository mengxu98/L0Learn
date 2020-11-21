#ifndef PARAMS_H
#define PARAMS_H
#include <map>
#include <RcppEigen.h>
#include "Model.h"

template <typename T>
struct Params {

    Model Specs;
    //std::string ModelType = "L0";
    std::vector<double> ModelParams {0, 0, 0, 2};
    std::size_t MaxIters = 500;
    double Tol = 1e-8;
    char Init = 'z';
    std::size_t RandomStartSize = 10;
    Eigen::SparseMatrix<double> * InitialSol;
    double b0 = 0; // intercept
    char CyclingOrder = 'c';
    std::vector<std::size_t> Uorder;
    bool ActiveSet = true;
    std::size_t ActiveSetNum = 6;
    std::size_t MaxNumSwaps = 200; // Used by CDSwaps
    std::vector<double> * Xtr;
    Eigen::VectorXd * ytX;
    std::map<std::size_t, Eigen::VectorXd> * D;
    std::size_t Iter = 0; // Current iteration number in the grid
    std::size_t ScreenSize = 1000;
   Eigen::VectorXd * r;
    T * Xy; // used for classification.
    std::size_t NoSelectK = 0;
    bool intercept = false;
   Eigen::VectorXd Lows;
   Eigen::VectorXd Highs;

};

#endif
