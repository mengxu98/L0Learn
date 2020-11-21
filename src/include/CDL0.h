#ifndef CDL0_H
#define CDL0_H
#include <tuple>
#include <RcppEigen.h>
#include "CD.h"
#include "Params.h"
#include "FitResult.h"
#include "utils.h"


template <class T>
class CDL0: public CD<T> {
    private:
        
       Eigen::VectorXd r; //vector of residuals
        
    public:
        CDL0(const T& Xi, const Eigen::VectorXd& yi, const Params<T>& P);
        //~CDL0(){}

        FitResult<T> Fit() final;
    
        inline double Objective(const Eigen::VectorXd &, const Eigen::SparseVector<double> &) final;
        
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
inline double CDL0<T>::GetBiGrad(const std::size_t i){
    return matrix_column_dot(*(this->X), i, this->r);
}

template <class T>
inline double CDL0<T>::GetBiValue(const double old_Bi, const double grd_Bi){
    return grd_Bi + old_Bi;
}

template <class T>
inline double CDL0<T>::GetBiReg(const double nrb_Bi){
    return std::abs(nrb_Bi);
}

// template <class T>
// inline double CDL0<T>::GetBiDelta(const double Bi_reg){
//     return std::sqrt(Bi_reg*Bi_reg - 2*this->lambda0);
// }

template <class T>
inline void CDL0<T>::ApplyNewBi(const std::size_t i, const double old_Bi, const double new_Bi){
    this->r += matrix_column_mult(*(this->X), i, old_Bi - new_Bi);
    this->B.coeffRef(i) = new_Bi;
}

template <class T>
inline void CDL0<T>::ApplyNewBiCWMinCheck(const std::size_t i, const double old_Bi, const double new_Bi){
    this->r += matrix_column_mult(*(this->X), i, old_Bi - new_Bi);
    this->B.coeffRef(i) = new_Bi;
}

template <class T>
inline double CDL0<T>::Objective(const Eigen::VectorXd & r, const Eigen::SparseVector<double> & B) {
    return 0.5 * r.dot(r) + this->lambda0 * B.nonZeros();
}

template <class T>
inline double CDL0<T>::Objective() {
    return 0.5 * this->r.dot(this->r) + this->lambda0 * this->B.nonZeros();
}



#endif
