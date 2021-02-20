#ifndef CDL012SWAPS_H
#define CDL012SWAPS_H
#include <map>
#include "RcppEigen.h"
#include "CD.h"
#include "CDSwaps.h"
#include "CDL012.h"
#include "FitResult.h"
#include "Params.h"
#include "utils.h"
#include "BetaVector.h"

template <class T>
class CDL012Swaps : public CDSwaps<T> { 
    public:
        CDL012Swaps(const T& Xi, const Eigen::ArrayXd& yi, const Params<T>& Pi);

        FitResult<T> _FitWithBounds() final;
        
        FitResult<T> _Fit() final;

        double Objective(const Eigen::ArrayXd & r, const beta_vector & B) final;
        
        double Objective() final;

};

template <class T>
inline double CDL012Swaps<T>::Objective(const Eigen::ArrayXd & r, const beta_vector & B) {
    const auto l2norm = B.norm();
    const auto l1norm = B.template lpNorm<1>();
    return 0.5 * r.square().sum() + this->lambda0 * n_nonzero(B) + this->lambda1 * l1norm + this->lambda2 * l2norm * l2norm;
}

template <class T>
inline double CDL012Swaps<T>::Objective() {
    throw std::runtime_error("CDL012Swaps does not have this->r.");
}

#endif
