#ifndef L0LEARN_UTILS_H
#define L0LEARN_UTILS_H
#include <vector>
#include "RcppEigen.h"
#include "BetaVector.h"

#include <chrono>
#include <thread>


template <typename T>
inline T clamp(T x, T low, T high) {
    // -O3 Compiler should remove branches
    if (x < low) 
        x = low;
    if (x > high) 
        x = high;
    return x;
}


std::vector<size_t> arg_sort_array(const Eigen::Ref<const Eigen::ArrayXd> arr,
                                   const std::string order = "descend");

void clamp_by_vector(Eigen::VectorXd &B, const Eigen::ArrayXd& lows, const Eigen::ArrayXd& highs);

void clamp_by_vector(Eigen::SparseVector<double> &B, const Eigen::ArrayXd& lows, const Eigen::ArrayXd& highs);


Eigen::ArrayXd inline make_predicitions(const Eigen::MatrixXd& X,
                                        const Eigen::SparseVector<double> &B,
                                        const double b0)
{
    return (X*B).array() + b0;

}

Eigen::ArrayXd inline make_predicitions(const Eigen::SparseMatrix<double>& X,
                                        const Eigen::SparseVector<double>& B,
                                        const double b0)
{
    return Eigen::MatrixXd(X*B).array() + b0;

}

template <typename T1>
Eigen::ArrayXd inline matrix_column_get(const Eigen::MatrixXd &mat, T1 col){
    return mat.col(col).array();
}

template <typename T1>
Eigen::ArrayXd inline matrix_column_get(const Eigen::SparseMatrix<double> &mat, T1 col){
    return Eigen::ArrayXd(mat.col(col));
}


template <typename mat_type, typename T1>
mat_type inline matrix_rows_get(const mat_type &mat, const T1 &vector_of_row_indices){
    // Option for CV for random splitting or contiguous splitting.
    // 1 - N without permutations splitting at floor(N/n_folds)
    mat_type row_mat(vector_of_row_indices.size(), mat.cols());
    
    for (auto i = 0; i < vector_of_row_indices.size(); i++){
        row_mat.row(i) = mat.row(vector_of_row_indices[i]);
    }
    return row_mat;
}

template <typename T1>
std::unordered_map<std::size_t, std::size_t> vector_indicies_dictionary(const T1 &vector_of_row_indices)
{
    std::unordered_map<std::size_t, std::size_t> vector_map;
    for (std::size_t i = 0; i < vector_of_row_indices.size(); i++)
    {
        vector_map.insert(std::pair<std::size_t, std::size_t>(vector_of_row_indices[i], i));
    }
    return vector_map;
}


template <typename T1>
Eigen::SparseMatrix<double> inline matrix_rows_get(const Eigen::SparseMatrix<double> &mat, const T1 &vector_of_row_indices){
    // Option for CV for random splitting or contiguous splitting.
    // 1 - N without permutations splitting at floor(N/n_folds)
    std::vector< Eigen::Triplet<double> > tripletList;
    std::size_t estimation_of_entries = mat.nonZeros() * vector_of_row_indices.size() / mat.rows();
    tripletList.reserve(estimation_of_entries);
    //Rcpp::Rcout << "1\n";
    //std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    auto vector_map = vector_indicies_dictionary(vector_of_row_indices);
    
    if (vector_map.size() != vector_of_row_indices.size()){
        Rcpp::stop("Does not support duplicate rows\n");
    }
    //Rcpp::Rcout << "2\n";
    //std::this_thread::sleep_for(std::chrono::milliseconds(100));
    for (std::size_t k = 0; k < mat.outerSize(); ++k){
        for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it){
            auto subset_row = vector_map.find(it.row());
            if (subset_row != vector_map.end()){
                tripletList.push_back(Eigen::Triplet<double>(subset_row->second,
                                                             it.col(),
                                                             it.value()));
            }
        }
    }
    //Rcpp::Rcout << "3\n";
    //std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    Eigen::SparseMatrix<double> row_mat(vector_of_row_indices.size(), mat.cols());
    
    //Rcpp::Rcout << row_mat.size() << " \n";
    //Rcpp::Rcout << "4\n";
    //std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // for (auto&x : tripletList){
    //     Rcpp::Rcout << x.row() << ", " <<  x.col() << ", " << x.value() << "\n";
    // }
    //std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    
    row_mat.setFromTriplets(tripletList.begin(), tripletList.end());
    row_mat.makeCompressed();
    
    //Rcpp::Rcout << "5\n";
    //std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return row_mat;
}

template <typename vec_type, typename index_type>
vec_type inline vector_subset(const vec_type &vec, const index_type &vector_of_indices){
    vec_type sub_vector;
    
    for (auto& ind: vector_of_indices){
        sub_vector.push_back(vec[ind]);
    }
    return sub_vector;
}

Eigen::RowVectorXd inline matrix_column_sums(const Eigen::MatrixXd& mat){
    return mat.colwise().sum();
}

Eigen::RowVectorXd inline matrix_column_sums(const Eigen::SparseMatrix<double>& mat){
    auto nrows = mat.rows();
    return Eigen::RowVectorXd::Ones(nrows) * mat;
}

template <typename T1, typename T2>
double inline matrix_column_dot(const Eigen::MatrixXd &mat, T1 col, const T2 &u){
    return matrix_column_get(mat, col).cwiseProduct(u).sum();
}

template <typename T1, typename T2>
double inline matrix_column_dot(const Eigen::SparseMatrix<double> &mat, T1 col, const T2 &u){
    return matrix_column_get(mat, col).cwiseProduct(u).sum();
}


Eigen::RowVectorXd matrix_normalize(Eigen::SparseMatrix<double> &mat_norm);

Eigen::RowVectorXd matrix_normalize(Eigen::MatrixXd &mat_norm);

Eigen::RowVectorXd matrix_center(const Eigen::MatrixXd& X, Eigen::MatrixXd& X_normalized, bool intercept);

Eigen::RowVectorXd matrix_center(const Eigen::SparseMatrix<double>& X, Eigen::SparseMatrix<double>& X_normalized, bool intercept);

#endif //L0LEARN_UTILS_H