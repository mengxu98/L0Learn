#include "Test_Interface.h" 
// [[Rcpp::depends(RcppArmadillo)]]


// [[Rcpp::export]]
Eigen::VectorXd R_matrix_column_get_dense(const Eigen::MatrixXd &mat, int col) {
    return matrix_column_get(mat, col);
}

// [[Rcpp::export]]
Eigen::VectorXd R_matrix_column_get_sparse(const Eigen::SparseMatrix<double> &mat, int col) {
    return matrix_column_get(mat, col);
}

// [[Rcpp::export]]
Eigen::MatrixXd R_matrix_rows_get_dense(const Eigen::MatrixXd &mat, const arma::ucolvec rows){
    return matrix_rows_get(mat, rows);
}
    
// [[Rcpp::export]]
Eigen::SparseMatrix<double> R_matrix_rows_get_sparse(const Eigen::SparseMatrix<double> &mat, const arma::ucolvec rows){
    return matrix_rows_get(mat, rows);
}

// [[Rcpp::export]]
Eigen::MatrixXd R_matrix_vector_schur_product_dense(const Eigen::MatrixXd &mat, constEigen::VectorXd &u){
    return matrix_vector_schur_product(mat, &u);
}

// [[Rcpp::export]]
Eigen::SparseMatrix<double> R_matrix_vector_schur_product_sparse(const Eigen::SparseMatrix<double> &mat, constEigen::VectorXd &u){
    return matrix_vector_schur_product(mat, &u);
}


// [[Rcpp::export]]
Eigen::MatrixXd R_matrix_vector_divide_dense(const Eigen::MatrixXd &mat, constEigen::VectorXd &u){
    return matrix_vector_divide(mat, u);
}

// [[Rcpp::export]]
Eigen::SparseMatrix<double> R_matrix_vector_divide_sparse(const Eigen::SparseMatrix<double> &mat, constEigen::VectorXd &u){
    return matrix_vector_divide(mat, u);
}

// [[Rcpp::export]]
Eigen::VectorXd R_matrix_column_sums_dense(const Eigen::MatrixXd &mat){
    return matrix_column_sums(mat);
}

// [[Rcpp::export]]
Eigen::VectorXd R_matrix_column_sums_sparse(const Eigen::SparseMatrix<double> &mat){
    return matrix_column_sums(mat);
}


// [[Rcpp::export]]
double R_matrix_column_dot_dense(const Eigen::MatrixXd &mat, int col, constEigen::VectorXd u){
    return matrix_column_dot(mat, col, u);
}

// [[Rcpp::export]]
double R_matrix_column_dot_sparse(const Eigen::SparseMatrix<double> &mat, int col, constEigen::VectorXd u){
    return matrix_column_dot(mat, col, u);
}

// [[Rcpp::export]]
Eigen::VectorXd R_matrix_column_mult_dense(const Eigen::MatrixXd &mat, int col, double u){
    return matrix_column_mult(mat, col, u);
}

// [[Rcpp::export]]
Eigen::VectorXd R_matrix_column_mult_sparse(const Eigen::SparseMatrix<double> &mat, int col, double u){
    return matrix_column_mult(mat, col, u);
}

// [[Rcpp::export]]
Rcpp::List R_matrix_normalize_dense(Eigen::MatrixXd mat_norm){
    Eigen::VectorXd ScaleX = matrix_normalize(mat_norm);
    return Rcpp::List::create(Rcpp::Named("mat_norm") = mat_norm,
                              Rcpp::Named("ScaleX") = ScaleX);
};

// [[Rcpp::export]]
Rcpp::List R_matrix_normalize_sparse(Eigen::SparseMatrix<double> mat_norm){
    Eigen::VectorXd ScaleX = matrix_normalize(mat_norm);
    return Rcpp::List::create(Rcpp::Named("mat_norm") = mat_norm,
                              Rcpp::Named("ScaleX") = ScaleX);
};

// [[Rcpp::export]]
Rcpp::List R_matrix_center_dense(Eigen::MatrixXd mat, bool intercept){
    auto martrix_center_return = matrix_center(mat, intercept);
    Eigen::MatrixXd X_normalized = std::get<0>(martrix_center_return);
    Eigen::VectorXd meanX = std::get<1>(martrix_center_return);
    return Rcpp::List::create(Rcpp::Named("mat_norm") = X_normalized,
                              Rcpp::Named("MeanX") = meanX);
};

// [[Rcpp::export]]
Rcpp::List R_matrix_center_sparse(Eigen::SparseMatrix<double> mat, bool intercept){
    auto martrix_center_return = matrix_center(mat, intercept);
    Eigen::SparseMatrix<double> X_normalized = std::get<0>(martrix_center_return);
    Eigen::VectorXd meanX = std::get<1>(martrix_center_return);
    
    return Rcpp::List::create(Rcpp::Named("mat_norm") = X_normalized,
                              Rcpp::Named("MeanX") = meanX);
};
