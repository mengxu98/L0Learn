#ifndef CDL0_H
#define CDL0_H
#include <tuple>
#include "RcppEigen.h"
#include "CD.h"
#include "Params.h"
#include "FitResult.h"
#include "utils.h"
#include "BetaVector.h"
#include "logging.h"

template <class T>
class CDL0: public CD<T, CDL0<T>>{
    private:
        
        Eigen::ArrayXd r; //vector of residuals
        
    public:
        CDL0(const T& Xi, const Eigen::ArrayXd& yi, const Params<T>& P);
        //~CDL0(){}

        FitResult<T> _FitWithBounds() final;
        
        FitResult<T> _Fit() final;
    
        inline double Objective(const Eigen::ArrayXd &, const beta_vector &) final;
        
        inline double Objective() final;
        
        inline double GetBiGrad(const std::size_t i);
        
        inline double GetBiValue(const double old_Bi, const double grd_Bi);
        
        inline double GetBiReg(const double nrb_Bi);
        
        inline void ApplyNewBi(const std::size_t i, const double old_Bi, const double new_Bi);
        
        inline void ApplyNewBiCWMinCheck(const std::size_t i, const double old_Bi, const double new_Bi);
};

template <class T>
inline double CDL0<T>::GetBiGrad(const std::size_t i){
    
    return matrix_column_get(*this->X, i).cwiseProduct(this->r).sum();
}

template <class T>
inline double CDL0<T>::GetBiValue(const double old_Bi, const double grd_Bi){
    return grd_Bi + old_Bi;
}

template <class T>
inline double CDL0<T>::GetBiReg(const double nrb_Bi){
    return std::abs(nrb_Bi);
}

template <class T>
inline void CDL0<T>::ApplyNewBi(const std::size_t i, const double old_Bi, const double new_Bi){
    
    //LOG("Beta Vector ApplyNewBi(" << i << "):  \n" << this->B << "\n");
    //LOG("Residual Vector ApplyNewBi(" << i << "): \n" << this->r << "\n");
    //LOG("X_i ApplyNewBi(" << i << "): \n" << matrix_column_get(*this->X, i) <<"\n");
    
    
    this->r += matrix_column_get(*this->X, i) * (old_Bi - new_Bi);
    this->B[i] = new_Bi;
}

template <class T>
inline void CDL0<T>::ApplyNewBiCWMinCheck(const std::size_t i, const double old_Bi, const double new_Bi){
    //LOG("Beta Vector ApplyNewBiCWMinCheck(" << i << "):  \n" << this->B << "\n");
    //LOG("Residual Vector ApplyNewBiCWMinCheck(" << i << "): \n" << this->r << "\n");
    //LOG("X_i ApplyNewBiCWMinCheck(" << i << "): \n" << matrix_column_get(*this->X, i) <<"\n");
    this->r += matrix_column_get(*this->X, i) * (old_Bi - new_Bi);
    this->B[i] = new_Bi;
    this->Order.push_back(i);
}

template <class T>
inline double CDL0<T>::Objective(const Eigen::ArrayXd & r, const beta_vector & B) {
    return 0.5 * r.square().sum() + this->lambda0 * n_nonzero(B);
}

template <class T>
inline double CDL0<T>::Objective() {
    LOG("Objective START");
    return 0.5 * this->r.square().sum() + this->lambda0 * n_nonzero(this->B);
}

template <class T>
CDL0<T>::CDL0(const T& Xi, const Eigen::ArrayXd& yi, const Params<T>& P) : CD<T, CDL0<T>>(Xi, yi, P){
    this->thr2 = 2 * this->lambda0;
    this->thr = sqrt(this->thr2); 
    this->r = P.r; 
    //this->result.r = P.r;
}

template <class T>
FitResult<T> CDL0<T>::_Fit() {
    LOG("Start");
    this->objective = Objective(this->r, this->B);
    //LOG("Beta Vector before Iterations: \n" << this->B << "\n");
    //LOG("Residual Vector before Iterations: \n" << this->r << "\n");
    //LOG("Objective before Iterations: \n" << this->objective <<"\n");
    
    std::vector<std::size_t> FullOrder = this->Order;
    
    if (this->ActiveSet) {
        this->Order.resize(std::min((int) (n_nonzero(this->B) + this->ScreenSize + this->NoSelectK), (int)(this->p))); // std::min(1000,Order.size())
    }
    
    
    
    LOG("Iters Start");
    for (std::size_t t = 0; t < this->MaxIters; ++t) {
        LOG("\nIter: " << t);
        this->Bprev = this->B;
        
        if (this->isSparse && this->intercept){
            this->UpdateSparse_b0(this->r);
        }
        
        LOG("this->Order: ");
        for (auto i = this->Order.begin(); i != this->Order.end(); ++i)
            LOG_DELAY(*i << ' ', 2);
        
        for (auto& i : this->Order) {
            this->UpdateBi(i);
        }
        
        LOG("RestrictSupport");
        this->RestrictSupport();
        
        if (this->isConverged() && this->CWMinCheck()) {
            break;
        }
    }
    LOG("Iters End");
    
    //LOG("Beta Vector after Iterations: \n" << this->B << "\n");
    //LOG("Residual Vector after Iterations: \n" << this->r << "\n");
    //LOG("Objective after Iterations:\n" << this->objective <<"\n");
    
    // Re-optimize b0 after convergence.
    if (this->isSparse && this->intercept){
        this->UpdateSparse_b0(this->r);
    }
    
    LOG("Results");
    this->result.Objective = this->objective;
    this->result.B = this->B;
    this->result.r = this->r; // change to pointer later
    this->result.IterNum = this->CurrentIters;
    this->result.b0 = this->b0;
    
    return this->result;
}

template <class T>
FitResult<T> CDL0<T>::_FitWithBounds() {
    clamp_by_vector(this->B, this->Lows, this->Highs);
    
    this->objective = Objective(this->r, this->B);
    
    std::vector<std::size_t> FullOrder = this->Order;
    
    if (this->ActiveSet) {
        this->Order.resize(std::min((int) (n_nonzero(this->B) + this->ScreenSize + this->NoSelectK), (int)(this->p))); // std::min(1000,Order.size())
    }
    
    for (std::size_t t = 0; t < this->MaxIters; ++t) {
        this->Bprev = this->B;
        
        if (this->isSparse && this->intercept){
            this->UpdateSparse_b0(this->r);
        }
        
        for (auto& i : this->Order) {
            this->UpdateBiWithBounds(i);
        }
        
        this->RestrictSupport();
        
        if (this->isConverged() && this->CWMinCheckWithBounds()) {
            break;
        }
    }
    
    // Re-optimize b0 after convergence.
    if (this->isSparse && this->intercept){
        this->UpdateSparse_b0(this->r);
    }
    
    this->result.Objective = this->objective;
    this->result.B = this->B;
    this->result.r = this->r; // change to pointer later
    this->result.IterNum = this->CurrentIters;
    this->result.b0 = this->b0;
    
    return this->result;
}

template class CDL0<Eigen::MatrixXd>;
template class CDL0<Eigen::SparseMatrix<double>>;

#endif
