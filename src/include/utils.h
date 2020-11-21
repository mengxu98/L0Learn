#ifndef L0LEARN_UTILS_H
#define L0LEARN_UTILS_H
#include <vector>
#include <RcppEigen.h>


template <typename T>
inline T clamp(T x, T low, T high) {
    // -O3 Compiler should remove branches
    if (x < low) 
        x = low;
    if (x > high) 
        x = high;
    return x;
}

template <typename T1>
Eigen::VectorXd inline matrix_column_get(const Eigen::MatrixXd &mat, T1 col){
    return mat.col(col);
}

template <typename T1>
Eigen::VectorXd inline matrix_column_get(const Eigen::SparseMatrix<double> &mat, T1 col){
    return Eigen::VectorXd(mat.col(col));
}

template <typename T1>
Eigen::MatrixXd inline matrix_rows_get(const Eigen::MatrixXd &mat, const T1 vector_of_row_indices){
    Eigen::MatrixXd mat_rows = Eigen::MatrixXd(vector_of_row_indices.size(), mat.cols());
    
    for (std::size_t i = 0; i < vector_of_row_indices.size(); ++i){
        mat_rows.row(i) = mat.row(vector_of_row_indices(i));
    }
    return mat_rows;
}

template <typename T1>
Eigen::SparseMatrix<double> inline matrix_rows_get(const Eigen::SparseMatrix<double> &mat, const T1 vector_of_row_indices){
    // Option for CV for random splitting or contiguous splitting.
    // 1 - N without permutations splitting at floor(N/n_folds)
    std::vector<Eigen::Triplet<double>> tripletList;
    
    auto vector_end = vector_of_row_indices.end();
    
    for (auto k=0; k<mat.outerSize(); ++k){
        for (Eigen::SparseMatrix<double>::InnerIterator it(mat,k); it; ++it) {
            
            auto old_row = it.row();
            auto new_row = find(vector_of_row_indices.begin(), vector_end, old_row);
        
            if (new_row != vector_end){
                // If element was found
                // k = it.col()
                tripletList.push_back(Eigen::Triplet<double>(old_row, k, it.value()));
            }
        }
    }
    
    Eigen::SparseMatrix<double> row_mat = Eigen::SparseMatrix<double>(vector_of_row_indices.n_elem, mat.cols());
    row_mat.setFromTriplets(tripletList.begin(), tripletList.end());

    return row_mat;
}

template <typename T1>
Eigen::MatrixXd inline matrix_vector_schur_product(const Eigen::MatrixXd &mat, const T1 &y){
    // return mat[i, j] * y[i] for each j
    return mat.cwiseProduct(y);
}

template <typename T1>
Eigen::SparseMatrix<double> inline matrix_vector_schur_product(const Eigen::SparseMatrix<double> &mat, const T1 &y){
    
    // Eigen::SparseMatrix<double> Xy = Eigen::SparseMatrix<double>(mat);
    // Eigen::SparseMatrix<double>::iterator begin = Xy.begin();
    // Eigen::SparseMatrix<double>::iterator end = Xy.end();
    // 
    // auto yp = (*y);
    // for (; begin != end; ++begin){
    //     auto row = begin.row();
    //     *begin = (*begin)  * yp(row);
    // }
    // return Xy;
    return mat.cwiseProduct(y);
}

template <typename T1>
Eigen::SparseMatrix<double> inline matrix_vector_divide(const Eigen::SparseMatrix<double>& mat, const T1 &u){
    // Eigen::SparseMatrix<double> divided_mat = Eigen::SparseMatrix<double>(mat);
    // 
    // //auto up = (*u);
    // Eigen::SparseMatrix<double>::iterator begin = divided_mat.begin();
    // Eigen::SparseMatrix<double>::iterator end = divided_mat.end();
    // for ( ; begin != end; ++begin){
    //     *begin = (*begin) / u(begin.row());
    // }
    // return divided_mat;
    return mat.cwiseQuotient(u);
}

template <typename T1>
Eigen::MatrixXd inline matrix_vector_divide(const Eigen::MatrixXd& mat, const T1 &u){
    return mat.cwiseQuotient(u);
}

Eigen::RowVectorXd inline matrix_column_sums(const Eigen::MatrixXd& mat){
    return mat.colwise().sum();
}

Eigen::RowVectorXd inline matrix_column_sums(const Eigen::SparseMatrix<double>& mat){
    return Eigen::RowVectorXd::Ones(mat.rows())*mat;
}

template <typename T1, typename T2>
double inline matrix_column_dot(const Eigen::MatrixXd &mat, T1 col, const T2 &u){
    return matrix_column_get(mat, col).dot(u);
}

template <typename T1, typename T2>
double inline matrix_column_dot(const Eigen::SparseMatrix<double> &mat, T1 col, const T2 &u){
    return matrix_column_get(mat, col).dot(u);
}

template <typename T1, typename T2>
Eigen::VectorXd inline matrix_column_mult(const Eigen::MatrixXd &mat, T1 col, const T2 &u){
    return matrix_column_get(mat, col).cwiseProduct(u);
}

template <typename T1, typename T2>
Eigen::VectorXd inline matrix_column_mult(const Eigen::SparseMatrix<double> &mat, T1 col, const T2 &u){
    return matrix_column_get(mat, col).cwiseProduct(u);
}

Eigen::VectorXd matrix_normalize(Eigen::SparseMatrix<double> &mat_norm);

Eigen::VectorXd matrix_normalize(Eigen::MatrixXd &mat_norm);

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> matrix_center(const Eigen::MatrixXd& X,
                                                  bool intercept);

std::tuple<Eigen::SparseMatrix<double>, Eigen::VectorXd> matrix_center(const Eigen::SparseMatrix<double>& X,
                                                     bool intercept);

Eigen::SparseMatrix<double> clamp_by_vector(Eigen::SparseMatrix<double>, const Eigen::VectorXd, const Eigen::VectorXd);

#endif //L0LEARN_UTILS_H
