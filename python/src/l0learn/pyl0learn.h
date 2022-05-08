#ifndef PYTHON_L0LEARN_CORE_H
#define PYTHON_L0LEARN_CORE_H

#include <carma>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "L0LearnCore.h"

struct sparse_mat {
  const arma::uvec rowind;
  const arma::uvec colptr;
  const arma::vec values;
  const arma::uword n_rows;
  const arma::uword n_cols;

  explicit sparse_mat(const arma::sp_mat &x) : rowind(arma::uvec(x.row_indices, x.n_elem)),
                                      colptr(arma::uvec(x.col_ptrs, x.n_cols + 1)),
                                      values(arma::vec(x.values, x.n_elem)),
                                      n_rows(x.n_rows),
                                      n_cols(x.n_cols)
                                          {}
  sparse_mat(const sparse_mat &x) : rowind(x.rowind),
                                      colptr(x.colptr),
                                      values(x.values),
                                      n_rows(x.n_rows),
                                      n_cols(x.n_cols)
  {}
  sparse_mat(const arma::uvec &rowind,
             const arma::uvec &colptr,
             const arma::vec &values,
             const arma::uword n_rows,
             const arma::uword n_cols) : rowind(rowind),
                                         colptr(colptr),
                                         values(values),
                                         n_rows(n_rows),
                                         n_cols(n_cols)
  {}

  arma::sp_mat to_arma_object() const {
    return {this->rowind, this->colptr, this->values, this->n_rows, this->n_cols};
  }

};

std::vector<arma::vec> field_to_vector(const arma::field<arma::vec>& x){
  std::vector<arma::vec> return_vector(x.n_elem);

  for (auto i=0; i < x.n_elem; i ++){
    return_vector[i] = x[i];
  }
  return return_vector;
}

template <class T1, class T2>
std::vector<T1> field_to_vector(const arma::field<T2>& x){
  std::vector<T1> return_vector;
  return_vector.reserve(x.n_elem);

  for (auto i=0; i < x.n_elem; i ++){
    return_vector.push_back(T1(x[i]));
  }
  return return_vector;
}

struct py_fitmodel {
  std::vector<std::vector<double>> Lambda0;
  std::vector<double> Lambda12;
  std::vector<std::vector<std::size_t>> NnzCount;
  std::vector<sparse_mat> Beta;
  std::vector<std::vector<double>> Intercept;
  std::vector<std::vector<bool>> Converged;

  py_fitmodel(const py_fitmodel &) = default;
  py_fitmodel(std::vector<std::vector<double>> &lambda0,
              std::vector<double> &lambda12,
              std::vector<std::vector<std::size_t>> &nnzCount,
              std::vector<sparse_mat> &beta,
              std::vector<std::vector<double>> &intercept,
              std::vector<std::vector<bool>> &converged)
      : Lambda0(lambda0),
        Lambda12(lambda12),
        NnzCount(nnzCount),
        Beta(beta),
        Intercept(intercept),
        Converged(converged) {}

  explicit py_fitmodel(const fitmodel & f) : Lambda0(f.Lambda0),
                                  Lambda12(f.Lambda12),
                                  NnzCount(f.NnzCount),
                                  Beta(field_to_vector<sparse_mat, arma::sp_mat>(f.Beta)),
                                  Intercept(f.Intercept),
                                  Converged(f.Converged) {}
};

struct py_cvfitmodel : py_fitmodel {
  std::vector<arma::vec> CVMeans;
  std::vector<arma::vec> CVSDs;

  py_cvfitmodel(const py_cvfitmodel &) = default;

  py_cvfitmodel(std::vector<std::vector<double>> &lambda0,
                std::vector<double> &lambda12,
                std::vector<std::vector<std::size_t>> &nnzCount,
                std::vector<sparse_mat> &beta,
                std::vector<std::vector<double>> &intercept,
                std::vector<std::vector<bool>> &converged,
                std::vector<arma::vec> &cVMeans,
                std::vector<arma::vec> &cVSDs)
      : py_fitmodel(lambda0,
                 lambda12,
                 nnzCount,
                 beta,
                 intercept,
                 converged),
        CVMeans(cVMeans), CVSDs(cVSDs) {}

  explicit py_cvfitmodel(const cvfitmodel & f) : py_fitmodel(f),
                                       CVMeans(field_to_vector(f.CVMeans)),
                                       CVSDs(field_to_vector(f.CVSDs)) {}

};

#endif //PYTHON_L0LEARN_CORE_H
