#ifndef BETA_VECTOR_H
#define BETA_VECTOR_H
#include <vector>
#include "RcppEigen.h"

/*
 * Eigen::VectorXd implementation
 */


using beta_vector = Eigen::VectorXd;
//using beta_vector = Eigen::SparseVector<double>;

std::vector<std::size_t> nnzIndicies(const Eigen::VectorXd& B, const std::size_t start=0);

std::vector<std::size_t> nnzIndicies(const Eigen::SparseVector<double>& B, const std::size_t start=0);

std::size_t inline n_nonzero(const Eigen::VectorXd& B){
    const auto nnzs = B.array() != 0;
    return nnzs.count();
    
}

std::size_t inline n_nonzero(const Eigen::SparseVector<double>& B){
    return B.nonZeros();
    
}

bool has_same_support(const Eigen::VectorXd& B1, const Eigen::VectorXd& B2);

bool has_same_support(const Eigen::SparseVector<double>& B1, const Eigen::SparseVector<double>& B2);


#endif // BETA_VECTOR_H