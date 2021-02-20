#include "Test_Interface.h" 
// [[Rcpp::depends(RcppEigen)]]


// [[Rcpp::export]]
Eigen::ArrayXd R_matrix_column_get_dense(const Eigen::MatrixXd &mat, int col) {
    return matrix_column_get(mat, col);
}

// [[Rcpp::export]]
Eigen::ArrayXd R_matrix_column_get_sparse(const Eigen::SparseMatrix<double> &mat, int col) {
    return matrix_column_get(mat, col);
}

// [[Rcpp::export]]
Eigen::MatrixXd R_matrix_rows_get_dense(const Eigen::MatrixXd &mat, const std::vector<std::size_t> rows){
    return matrix_rows_get(mat, rows);
}
    
// [[Rcpp::export]]
Eigen::SparseMatrix<double> R_matrix_rows_get_sparse(const Eigen::SparseMatrix<double> &mat, const std::vector<std::size_t> rows){
    return matrix_rows_get(mat, rows);
}


// [[Rcpp::export]]
Eigen::Array<double, 1, Eigen::Dynamic> R_matrix_column_sums_dense(const Eigen::MatrixXd &mat){
    return matrix_column_sums(mat);
}

// [[Rcpp::export]]
Eigen::Array<double, 1, Eigen::Dynamic> R_matrix_column_sums_sparse(const Eigen::SparseMatrix<double> &mat){
    return matrix_column_sums(mat);
}


// [[Rcpp::export]]
double R_matrix_column_dot_dense(const Eigen::MatrixXd &mat, int col, const Eigen::ArrayXd u){
    return matrix_column_dot(mat, col, u);
}

// [[Rcpp::export]]
double R_matrix_column_dot_sparse(const Eigen::SparseMatrix<double> &mat, int col, const Eigen::ArrayXd u){
    return matrix_column_dot(mat, col, u);
}


// [[Rcpp::export]]
Rcpp::List R_matrix_normalize_dense(Eigen::MatrixXd mat_norm){
    Eigen::Array<double, 1, Eigen::Dynamic> ScaleX = matrix_normalize(mat_norm);
    return Rcpp::List::create(Rcpp::Named("mat_norm") = mat_norm,
                              Rcpp::Named("ScaleX") = ScaleX);
};

// [[Rcpp::export]]
Rcpp::List R_matrix_normalize_sparse(Eigen::SparseMatrix<double> mat_norm){
    Eigen::Array<double, 1, Eigen::Dynamic> ScaleX = matrix_normalize(mat_norm);
    return Rcpp::List::create(Rcpp::Named("mat_norm") = mat_norm,
                              Rcpp::Named("ScaleX") = ScaleX);
};

// [[Rcpp::export]]
Rcpp::List R_matrix_center_dense(const Eigen::MatrixXd mat, Eigen::MatrixXd X_normalized, bool intercept){
    Eigen::Array<double, 1, Eigen::Dynamic> meanX = matrix_center(mat, X_normalized, intercept);
    return Rcpp::List::create(Rcpp::Named("mat_norm") = X_normalized,
                              Rcpp::Named("MeanX") = meanX);
};

// [[Rcpp::export]]
Rcpp::List R_matrix_center_sparse(const Eigen::SparseMatrix<double> mat, Eigen::SparseMatrix<double> X_normalized,bool intercept){
    Eigen::Array<double, 1, Eigen::Dynamic> meanX = matrix_center(mat, X_normalized, intercept);
    return Rcpp::List::create(Rcpp::Named("mat_norm") = X_normalized,
                              Rcpp::Named("MeanX") = meanX);
};

// [[Rcpp::export]]
std::vector<size_t> R_vector_subset(const std::vector<size_t> x, const std::vector<size_t> indicies){
    return vector_subset(x, indicies);
}

// [[Rcpp::export]]
std::vector<size_t> R_arg_sort(const Eigen::ArrayXd&x)
{
    return arg_sort_array(x);
}

// [[Rcpp::export]]
Eigen::ArrayXd R_make_predicitions_dense(const Eigen::MatrixXd& X,
                                         const Eigen::VectorXd &B,
                                         const double b0)
{
    const Eigen::SparseVector<double> B_sparse = B.sparseView();
    return make_predicitions(X, B_sparse, b0);
}

// [[Rcpp::export]]
Eigen::ArrayXd R_make_predicitions_sparse(const Eigen::SparseMatrix<double>& X,
                                          const Eigen::VectorXd &B,
                                          const double b0)
{
    const Eigen::SparseVector<double> B_sparse = B.sparseView();
    return make_predicitions(X, B_sparse, b0);
}
    
    