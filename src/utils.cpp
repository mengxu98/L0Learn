#include "utils.h"


void clamp_by_vector(Eigen::VectorXd &B, const Eigen::ArrayXd& lows, const Eigen::ArrayXd& highs){
    const std::size_t n = B.rows();
    for (std::size_t i = 0; i < n; i++){
        B[i] = clamp(B.coeff(i), lows.coeff(i), highs.coeff(i));
    }
}

// void clamp_by_vector2(const Eigen::Ref<const Eigen::VectorXd> Bin,
//                       const Eigen::Ref<const Eigen::ArrayXd> lows,
//                       const Eigen::Ref<const Eigen::ArrayXd> highs,
//                       Eigen::Ref<Eigen::VectorXd> Bclamp)
// {
//     Bclamp = Bin.array().cwiseMin(highs).cwiseMax(lows).matrix();
// }

void clamp_by_vector(Eigen::SparseVector<double> &B, const Eigen::ArrayXd& lows, const Eigen::ArrayXd& highs){
    // See above implementation without filter for error.
    for (Eigen::SparseVector<double>::InnerIterator it(B); it; ++it) { 
        double B_item = it.value();
        const double low = lows[it.index()];
        const double high = highs[it.index()];
        it.valueRef() = clamp(B_item, low, high);
    }
}

Eigen::RowVectorXd matrix_normalize(Eigen::SparseMatrix<double> &mat_norm){
    // TODO: Numerical Instabilities should be handled in R?
    auto p = mat_norm.cols();
    auto n = mat_norm.rows();
    Eigen::RowVectorXd scaleX = Eigen::RowVectorXd::Zero(p); // will contain the l2norm of every col
    Eigen::ArrayXd ones = Eigen::ArrayXd::Ones(n);
    for (auto col = 0; col < p; col++){
        double l2norm = mat_norm.col(col).norm();
        if (l2norm == 0) 
        {
            l2norm = 1;
        } else if (l2norm != l2norm) // Nan Check
        {
            Rcpp::stop("Convergence Error: Nan exsts in X");
        }

        scaleX(col) = l2norm;
        
        mat_norm.col(col) = mat_norm.col(col)/l2norm;
    }
    
    return scaleX;
}

Eigen::RowVectorXd matrix_normalize(Eigen::MatrixXd& mat_norm){

    auto p = mat_norm.cols();
    Eigen::RowVectorXd scaleX = Eigen::RowVectorXd::Zero(p); // will contain the l2norm of every col
    
    for (auto col = 0; col < p; col++) {
        double l2norm = mat_norm.col(col).norm();
        if (l2norm == 0) 
        {
            l2norm = 1;
        } else if (l2norm != l2norm) // Nan Check
        {
            Rcpp::stop("Convergence Error: Nan exsts in X");
        }
        
        scaleX(col) = l2norm;
    }
    
    mat_norm.array().rowwise() /= scaleX.array();
    
    return scaleX;
}

Eigen::RowVectorXd matrix_center(const Eigen::MatrixXd& X,
                                                      Eigen::MatrixXd& X_normalized, 
                                                      bool intercept){
    auto p = X.cols();
    Eigen::RowVectorXd meanX;
    
    if (intercept){
        meanX = X.colwise().mean();
        X_normalized = X.rowwise() - meanX;
    } else {
        meanX = Eigen::RowVectorXd::Zero(p);
        X_normalized = Eigen::MatrixXd(X);
    }
    
    return meanX;
}

Eigen::RowVectorXd matrix_center(const Eigen::SparseMatrix<double>& X, 
                                                      Eigen::SparseMatrix<double>& X_normalized, 
                                                      bool intercept){
    auto p = X.cols();
    Eigen::Array<double, Eigen::Dynamic, 1> meanX = Eigen::Array<double, Eigen::Dynamic, 1>::Zero(p);
    X_normalized = Eigen::SparseMatrix<double>(X);
    return meanX;
}

std::vector<size_t> arg_sort_array(const Eigen::Ref<const Eigen::ArrayXd> arr,
                                   const std::string order)
{
    auto n = arr.rows();
    
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    
    if (order.compare("descend") == 0){
        std::sort(idx.begin(), idx.end(), [&arr](const std::size_t &a, const std::size_t &b)
        {return arr[a] > arr[b];});
    } else {
        // Ascend
        std::sort(idx.begin(), idx.end(), [&arr](const std::size_t &a, const std::size_t &b)
        {return arr[a] > arr[b];});
    }
    
    return idx;
}
