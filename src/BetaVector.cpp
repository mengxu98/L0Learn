#include "BetaVector.h"

/*
 * Eigen::VectorXd implementation
 */

std::vector<std::size_t> nnzIndicies(const Eigen::VectorXd& B, const std::size_t start){
    // Returns a vector of the Non Zero Indicies of B
    std::vector<std::size_t> S;
    std::size_t nrows = B.rows();
    for (std::size_t i = start; i <= nrows; i++){
        if (B.coeff(i) != 0){
            S.push_back(i);
        }
    }
    return S;
}

std::vector<std::size_t> nnzIndicies(const Eigen::SparseVector<double>& B, const std::size_t start){
    // Returns a vector of the Non Zero Indicies of B
    std::vector<std::size_t> S;
    for (Eigen::SparseVector<double>::InnerIterator it(B); it; ++it)
    {
        if (it.index() >= start)
        {
            S.push_back(it.index());    
        }
    }
    return S;
}


std::size_t n_nonzero(const Eigen::VectorXd& B){
    const auto nnzs = (B.array() != 0);
    return nnzs.count();
    
}

std::size_t n_nonzero(const  Eigen::SparseVector<double>& B){
    return B.nonZeros();
    
}

bool has_same_support(const Eigen::VectorXd& B1, const Eigen::VectorXd& B2){
    if (B1.rows() != B2.rows()){
        return false;
    }
    std::size_t n = B1.rows();
    
    bool same_support = true;
    for (std::size_t i = 0; i < n; i++){
        same_support = same_support && ((B1.coeff(i) != 0) == (B2.coeff(i) != 0));
    }
    return same_support;
}

bool has_same_support(const Eigen::SparseVector<double>& B1, const  Eigen::SparseVector<double>& B2){
    
    if (B1.nonZeros() != B2.nonZeros()) {
        return false;
    } else {  // same number of nnz and Supp is sorted

        Eigen::SparseVector<double>::InnerIterator it1(B1);
        Eigen::SparseVector<double>::InnerIterator it2(B2);
        
        for(; it1 ; ++it1, ++it2)
        {
            if(it1.index() != it2.index())
            {
                return false;
            }
        }
        return true;
    }
}

