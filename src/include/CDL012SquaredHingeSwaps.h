#ifndef CDL012SquredHingeSwaps_H
#define CDL012SquredHingeSwaps_H
#include "RcppEigen.h"
#include "CD.h"
#include "CDSwaps.h"
#include "CDL012SquaredHinge.h"
#include "Params.h"
#include "FitResult.h"
#include "utils.h"
#include "BetaVector.h"

template <class T>
class CDL012SquaredHingeSwaps : public CDSwaps<T> {
    private:
        const double LipschitzConst = 2;
        double twolambda2;
        double qp2lamda2;
        double lambda1ol;
        double stl0Lc;
        // std::vector<double> * Xtr;
        
    public:
        CDL012SquaredHingeSwaps(const T& Xi, const Eigen::ArrayXd& yi, const Params<T>& P);

        FitResult<T> _FitWithBounds() final;
        
        FitResult<T> _Fit() final;

        inline double Objective(const Eigen::ArrayXd& r, const beta_vector & B) final;
        
        inline double Objective() final;


};

template <class T>
inline double CDL012SquaredHingeSwaps<T>::Objective(const Eigen::ArrayXd & onemyxb, const beta_vector & B) {
    const auto l2norm = B.norm();
    const auto l1norm = B.template lpNorm<1>();
    return onemyxb.cwiseMax(0).square().sum() + this->lambda0 * n_nonzero(B) + this->lambda1 * l1norm + this->lambda2 * l2norm * l2norm;
}

template <class T>
inline double CDL012SquaredHingeSwaps<T>::Objective() {
    throw std::runtime_error("CDL012SquaredHingeSwaps does not have this->onemyxb");
}


#endif
