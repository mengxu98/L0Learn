#ifndef RINTERFACE_H
#define RINTERFACE_H

#include <vector>
#include <string>
#include "RcppEigen.h"
#include "Grid.h"
#include "GridParams.h"
#include "FitResult.h"
#include "logging.h"


inline void to_arma_error() {
    Rcpp::stop("L0Learn.fit only supports sparse matricies (dgCMatrix), 2D arrays (Dense Matricies)");
}

#endif // RINTERFACE_H
