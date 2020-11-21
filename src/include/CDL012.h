#ifndef CDL012_H
#define CDL012_H
#include <RcppEigen.h>
#include "CD.h"
#include "FitResult.h"
#include "utils.h"

template <class T>
class CDL012 : public CD<T> {
    private:
        double Onep2lamda2;
       Eigen::VectorXd r; //vector of residuals
        
    public:
        CDL012(const T& Xi, const Eigen::VectorXd& yi, const Params<T>& P);
        //~CDL012(){}

        FitResult<T> Fit() final;

        inline double Objective(const Eigen::VectorXd & r, const Eigen::SparseVector<double> & B) final;
        
        inline double Objective() final;

        inline double GetBiGrad(const std::size_t i) final;
        
        inline double GetBiValue(const double old_Bi, const double grd_Bi) final;
        
        inline double GetBiReg(const double nrb_Bi) final;
        
        // inline double GetBiDelta(const double reg_Bi) final;
        
        inline void ApplyNewBi(const std::size_t i, const double old_Bi, const double new_Bi) final; 
        
        inline void ApplyNewBiCWMinCheck(const std::size_t i, const double old_Bi, const double new_Bi) final;
        
        bool CWMinCheck();

};


template <class T>
inline double CDL012<T>::GetBiGrad(const std::size_t i){
    return matrix_column_dot(*(this->X), i, this->r);
}

template <class T>
inline double CDL012<T>::GetBiValue(const double old_Bi, const double grd_Bi){
    return grd_Bi + old_Bi;
}

template <class T>
inline double CDL012<T>::GetBiReg(const double nrb_Bi){
    // sign(nrb_Bi)*(|nrb_Bi| - lambda1)/(1 + 2*lambda2)
    return (std::abs(nrb_Bi) - this->lambda1) / Onep2lamda2;
}

// template <class T>
// inline double CDL012<T>::GetBiDelta(const double Bi_reg){
//     return std::sqrt(Bi_reg*Bi_reg - this->thr2);
// }

template <class T>
inline void CDL012<T>::ApplyNewBi(const std::size_t i, const double Bi_old, const double Bi_new){
    this->r += matrix_column_mult(*(this->X), i, Bi_old - Bi_new);
    this->B.coeffRef(i) = Bi_new;
}

template <class T>
inline void CDL012<T>::ApplyNewBiCWMinCheck(const std::size_t i, const double Bi_old, const double Bi_new){
    this->r += matrix_column_mult(*(this->X), i, Bi_old - Bi_new);
    this->B.coeffRef(i) = Bi_new;
}

template <class T>
inline double CDL012<T>::Objective(const Eigen::VectorXd & r, const Eigen::SparseMatrix<double> & B) { 
    auto l2norm = B.norm();
    return 0.5 * r.dot(r) + this->lambda0 * B.nonZeros() + this->lambda1 * B.lpNorm() + this->lambda2 * l2norm * l2norm;
}

template <class T>
inline double CDL012<T>::Objective() { 
    auto l2norm = this->B.norm();
    return 0.5 * this->r.dot(this->r) + this->lambda0 * this->B.nonZeros() + this->lambda1 * this->B.lpNorm() + this->lambda2 * l2norm * l2norm;
}

#endif
