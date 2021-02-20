#ifndef CDL012SquaredHinge_H
#define CDL012SquaredHinge_H
#include "RcppEigen.h"
#include "CD.h"
#include "FitResult.h"
#include "Params.h"
#include "utils.h"
#include "BetaVector.h"

template <class T>
class CDL012SquaredHinge : public CD<T, CDL012SquaredHinge<T>> {
    private:
        const double LipschitzConst = 2; // for f (without regularization)
        double twolambda2;
        double qp2lamda2;
        double lambda1ol;
        // std::vector<double> * Xtr;
        Eigen::ArrayXd onemyxb;
        Eigen::Array<std::size_t, Eigen::Dynamic, 1> indices;
        T * Xy;


    public:
        CDL012SquaredHinge(const T& Xi, const Eigen::ArrayXd& yi, const Params<T>& P);
        
        //~CDL012SquaredHinge(){}

        FitResult<T> _FitWithBounds() final;
        
        FitResult<T> _Fit() final;

        inline double Objective(const Eigen::ArrayXd& r, const beta_vector & B) final;
        
        inline double Objective() final;
        
        inline double GetBiGrad(const std::size_t i);
        
        inline double GetBiValue(const double old_Bi, const double grd_Bi);
        
        inline double GetBiReg(const double Bi_step);
        
        inline void ApplyNewBi(const std::size_t i, const double Bi_old, const double Bi_new);
        
        inline void ApplyNewBiCWMinCheck(const std::size_t i, const double old_Bi, const double new_Bi);
        
};

template <class T>
inline double CDL012SquaredHinge<T>::GetBiGrad(const std::size_t i){
    // Rcpp::Rcout << "Grad stuff: " << arma::sum(2 * onemyxb.elem(indices) % (- matrix_column_get(*Xy, i).elem(indices))  ) << "\n";
    //return arma::sum(2 * onemyxb.elem(indices) % (- matrix_column_get(*Xy, i).elem(indices)) ) + twolambda2 * this->B[i];
    return 2*(onemyxb.max(0)*-matrix_column_get(*Xy, i)).sum() + twolambda2 * this->B[i];
}

template <class T>
inline double CDL012SquaredHinge<T>::GetBiValue(const double old_Bi, const double grd_Bi){
    return old_Bi - grd_Bi / qp2lamda2;
}

template <class T>
inline double CDL012SquaredHinge<T>::GetBiReg(const double Bi_step){
    return std::abs(Bi_step) - lambda1ol;
}

template <class T>
inline void CDL012SquaredHinge<T>::ApplyNewBi(const std::size_t i, const double Bi_old, const double Bi_new){
    onemyxb += (Bi_old - Bi_new) * matrix_column_get(*this->Xy, i);
    this->B[i] = Bi_new;
}

template <class T>
inline void CDL012SquaredHinge<T>::ApplyNewBiCWMinCheck(const std::size_t i, const double Bi_old, const double Bi_new){
    onemyxb += (Bi_old - Bi_new) * matrix_column_get(*this->Xy, i);
    this->B[i] = Bi_new;
    this->Order.push_back(i);
}

template <class T>
inline double CDL012SquaredHinge<T>::Objective(const Eigen::ArrayXd& onemyxb, const beta_vector & B) {
    const auto l2norm = B.norm();
    const auto l1norm = B.template lpNorm<1>();
    return onemyxb.cwiseMax(0).square().sum() + this->lambda0 * n_nonzero(B) + this->lambda1 * l1norm + this->lambda2 * l2norm * l2norm;
}


template <class T>
inline double CDL012SquaredHinge<T>::Objective() {
    
    const auto l2norm = this->B.norm();
    const auto l1norm = this->B.template lpNorm<1>();
    return onemyxb.max(0).square().sum() + this->lambda0 * n_nonzero(this->B) + this->lambda1 *l1norm + this->lambda2 * l2norm * l2norm;
}

template <class T>
CDL012SquaredHinge<T>::CDL012SquaredHinge(const T& Xi, const Eigen::ArrayXd& yi, const Params<T>& P) : CD<T, CDL012SquaredHinge<T>>(Xi, yi, P) {
    twolambda2 = 2 * this->lambda2;
    qp2lamda2 = (LipschitzConst + twolambda2); // this is the univariate lipschitz const of the differentiable objective
    this->thr2 = (2 * this->lambda0) / qp2lamda2;
    this->thr = std::sqrt(this->thr2);
    lambda1ol = this->lambda1 / qp2lamda2;
    
    // TODO: Review this line
    // TODO: Pass work from previous solution.
    auto XB = (*this->X)*this->B.matrix();
    onemyxb = 1 - (XB.array() + this->b0).cwiseProduct(this->y);
    
    Xy = P.Xy;
}

template <class T>
FitResult<T> CDL012SquaredHinge<T>::_Fit() {
    
    this->objective = Objective(); // Implicitly uses onemyx
    
    
    std::vector<std::size_t> FullOrder = this->Order; // never used in LR
    this->Order.resize(std::min((int) (n_nonzero(this->B) + this->ScreenSize + this->NoSelectK), (int)(this->p)));
    
    
    for (auto t = 0; t < this->MaxIters; ++t) {
        
        this->Bprev = this->B;
        
        // Update the intercept
        if (this->intercept) {
            const double b0old = this->b0;
            const double partial_b0 = 2*(onemyxb.max(0) *-this->y.max(0)).sum();
            this->b0 -= partial_b0 / (this->n * LipschitzConst); // intercept is not regularized
            onemyxb += this->y * (b0old - this->b0);
        }
        
        for (auto& i : this->Order) {
            this->UpdateBi(i);
        }
        
        this->RestrictSupport();
        
        // only way to terminate is by (i) converging on active set and (ii) CWMinCheck
        if ((this->isConverged()) && this->CWMinCheck()) {
            break;
        }
    }
    
    this->result.Objective = this->objective;
    this->result.B = this->B;
    this->result.Model = this;
    this->result.b0 = this->b0;
    this->result.IterNum = this->CurrentIters;
    this->result.onemyxb = this->onemyxb;
    return this->result;
}


template <class T>
FitResult<T> CDL012SquaredHinge<T>::_FitWithBounds() {
    
    clamp_by_vector(this->B, this->Lows, this->Highs);
    
    this->objective = Objective(); // Implicitly uses onemyx
    
    std::vector<std::size_t> FullOrder = this->Order; // never used in LR
    this->Order.resize(std::min((int) (n_nonzero(this->B) + this->ScreenSize + this->NoSelectK), (int)(this->p)));
    
    
    for (auto t = 0; t < this->MaxIters; ++t) {
        
        this->Bprev = this->B;
        
        // Update the intercept
        if (this->intercept) {
            const double b0old = this->b0;
            const double partial_b0 = 2*(onemyxb.max(0)*-this->y.max(0)).sum();
            this->b0 -= partial_b0 / (this->n * LipschitzConst); // intercept is not regularized
            onemyxb += this->y * (b0old - this->b0);
        }
        
        for (auto& i : this->Order) {
            this->UpdateBiWithBounds(i);
        }
        
        this->RestrictSupport();
        
        // only way to terminate is by (i) converging on active set and (ii) CWMinCheck
        if (this->isConverged()) {
            if (this->CWMinCheckWithBounds()) {
                break;
            }
        }
    }
    
    this->result.Objective = this->objective;
    this->result.B = this->B;
    this->result.Model = this;
    this->result.b0 = this->b0;
    this->result.IterNum = this->CurrentIters;
    this->result.onemyxb = this->onemyxb;
    return this->result;
}

template class CDL012SquaredHinge<Eigen::MatrixXd>;
template class CDL012SquaredHinge<Eigen::SparseMatrix<double>>;

#endif
