#include "utils.h"

// Eigen::SparseMatrix<double> clamp_by_vector(Eigen::SparseMatrix<double> B, constEigen::VectorXd lows, constEigen::VectorXd highs){
//     // Somehow this implementation fails unexpectedly.
//     // Calling
//     // >fit <- L0Learn.fit(X, y, algorithm = "CDPSI", lows=0)
//     // ...
//     // Clamp(-0.0379001, indx 859, indx 859)
//     // Clamp(-0.0379001, 0, inf)
//     // Clamp(0.196179, indx 936, indx 936)
//     // Clamp(0.196179, 0, inf)
//     // Clamp(0, indx 0, indx 0)
//     // Clamp(0, 0, inf)
//     // Clamp(-3.10504e+231, indx 3774873619, indx 3774873619)
//     // Clamp(-3.10504e+231, 
//     //       error: Mat::operator(): index out of bounds
//     //           Error in L0Learn.fit(X, y, algorithm = "CDPSI", lows = 0) : 
//     //           Mat::operator(): index out of bounds
//     Rcpp::Rcout << "Begin Clamp\n";
//     Rcpp::Rcout << "B Size" << B.size() << "\n";
//     Rcpp::Rcout << B << "\n";
//     Rcpp::Rcout << "lows Size" << lows.size() << "\n";
//     Rcpp::Rcout << "highs Size" << highs.size() << "\n";
//     auto begin = B.begin();
//     auto end = B.end();
//     for (; begin != end; ++begin){
//         double v = *begin;
//         Rcpp::Rcout << "Clamp(" << v << ", indx " << begin.row() << ", indx " << begin.row() << ")\n";
//         Rcpp::Rcout << "Clamp(" << v << ", " << lows(begin.row()) << ", " << highs(begin.row()) << ")\n";
//         double d = clamp(v,  lows(begin.row()), highs(begin.row()));
//         *begin = d;
//     }
//     Rcpp::Rcout << "End Clamp\n";
//     return B;
// }

Eigen::SparseMatrix<double> clamp_by_vector(Eigen::SparseMatrix<double> B, constEigen::VectorXd lows, constEigen::VectorXd highs){
    // See above implementation without filter for error.
    return B.max(lows).min(highs);
}

Eigen::RowVectorXd matrix_normalize(Eigen::SparseMatrix<double> &mat_norm){
    Eigen::VectorXd scaleX = mat_norm.colwise().norm();
    
    mat_norm.colwise().normalize(); // Inplace Operations
    
    return scaleX;
}

Eigen::RowVectorXd matrix_normalize(Eigen::MatrixXd& mat_norm){
    Eigen::VectorXd scaleX = mat_norm.colwise().norm();
    
    mat_norm.colwise().normalize(); // Inplace Operations
    
    return scaleX;
}

std::tuple<Eigen::MatrixXd, Eigen::RowVectorXd> matrix_center(const Eigen::MatrixXd& X, bool intercept){
    auto p = X.n_cols;
    Eigen::RowVectorXd meanX;
    Eigen::MatrixXd X_normalized;
    
    if (intercept){
        meanX = X.colwise().mean();
        X_normalized = X.rowwise() - meanX;
    } else {
        meanX = Eigen::RowVectorXd::Zero(p);
        X_normalized = Eigen::MatrixXd(X);
    }
    
    return std::make_tuple(X_normalized, meanX);
}

std::tuple<Eigen::SparseMatrix<double>, Eigen::RowVectorXd> matrix_center(const Eigen::SparseMatrix<double>& X,
                                                     bool intercept){
    auto p = X.n_cols;
    Eigen::RowVectorXd meanX = meanX = Eigen::RowVectorXd::Zero(p);;
    Eigen::SparseMatrix<double> X_normalized = Eigen::SparseMatrix<double>(X);
    return std::make_tuple(X_normalized, meanX);
}
