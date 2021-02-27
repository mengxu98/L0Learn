#ifndef CD_H
#define CD_H
#include <algorithm>
#include "RcppEigen.h"
#include "BetaVector.h"
#include "FitResult.h"
#include "Params.h"
#include "Model.h"
#include "utils.h"
#include "logging.h"

constexpr double lambda1_fudge_factor = 1e-15;


template<class T>
class CDBase {
    protected:
        std::size_t NoSelectK;
        std::vector<double> Xtr;
        std::size_t n, p;
        std::size_t Iter;
    
        beta_vector B;
        beta_vector Bprev;
        
        std::size_t SameSuppCounter = 0;
        double objective;
        std::vector<std::size_t> Order; // Cycling order
        std::vector<std::size_t> OldOrder; // Cycling order to be used after support stabilization + convergence.
        FitResult<T> result;
        
        /* Intercept and b0 are used for:
         *  1. Classification as b0 is updated iteratively in the CD algorithm 
         *  2. Regression on Sparse Matrices as we cannot adjust the support of 
         *  the columns of X and thus b0 must be updated iteraveily from the 
         *  residuals 
         */ 
        double b0 = 0; 
        double lambda1;
        double lambda0;
        double lambda2;
        double thr;
        double thr2; // threshold squared; 
        
        bool isSparse;
        bool intercept; 
        bool withBounds;

    public:
        const T * X;
        const Eigen::ArrayXd y;
        std::vector<double> ModelParams;

        char CyclingOrder;
        std::size_t MaxIters;
        std::size_t CurrentIters; // current number of iterations - maintained by Converged()
        const double rtol;
        const double atol;
        bool ActiveSet;
        std::size_t ActiveSetNum;
        bool Stabilized = false;
        const Eigen::ArrayXd Lows;
        const Eigen::ArrayXd Highs;


        CDBase(const T& Xi, const Eigen::ArrayXd& yi, const Params<T>& P);
        
        FitResult<T> Fit();

        virtual ~CDBase(){}

        virtual inline double Objective(const Eigen::ArrayXd&, const beta_vector&)=0;
        
        virtual inline double Objective()=0;
        
        virtual FitResult<T> _FitWithBounds() = 0;
        
        virtual FitResult<T> _Fit() = 0;

        static CDBase * make_CD(const T& Xi, const Eigen::ArrayXd& yi, const Params<T>& P);
        
};

template<class T>
class CDSwaps : public CDBase<T> {
    
protected:
    std::size_t MaxNumSwaps;
    Params<T> P;
    
public:
    
    CDSwaps(const T& Xi, const Eigen::ArrayXd& yi, const Params<T>& P);
    
    virtual ~CDSwaps(){};
    
};

class NotImplemented : public std::logic_error{
    public:
        NotImplemented() : std::logic_error("Function not yet implemented") { };
};

template<class T, class Derived>
class CD : public CDBase<T>{
    protected:
        std::size_t ScreenSize;
        std::vector<std::size_t> Range1p;
        
    public:
        CD(const T& Xi, const Eigen::ArrayXd& yi, const Params<T>& P);
        
        virtual ~CD(){};
        
        inline double GetBiGrad(const std::size_t i){
            // Must be implemented in Child Classes
            throw NotImplemented();
        };
        
        inline double GetBiValue(const double old_Bi, const double grd_Bi){
            // Must be implemented in Child Classes
            throw NotImplemented();
        };
        
        inline double GetBiReg(const double nrb_Bi){
            // Must be implemented in Child Classes
            throw NotImplemented();
        };
        
        inline void ApplyNewBi(const std::size_t i, const double old_Bi, const double new_Bi){
            // Must be implemented in Child Classes
            throw NotImplemented();
        };
        
        inline void ApplyNewBiCWMinCheck(const std::size_t i, const double old_Bi, const double new_Bi){
            // Must be implemented in Child Classes
            throw NotImplemented();
        };
        
        void UpdateBi(const std::size_t i);
        
        void UpdateBiWithBounds(const std::size_t i);
        
        bool UpdateBiCWMinCheck(const std::size_t i, const bool Cwmin);
        
        bool UpdateBiCWMinCheckWithBounds(const std::size_t i, const bool Cwmin);
        
        void RestrictSupport();
        
        void UpdateSparse_b0(Eigen::ArrayXd &r);
        
        bool isConverged();
        
        bool CWMinCheck();
        
        bool CWMinCheckWithBounds();
        
};


template<class T, class Derived>
void CD<T, Derived>::UpdateBiWithBounds(const std::size_t i){
    // Update a single coefficient of B for various CD Settings
    // The following functions are virtual and must be defined for any CD implementation.
    //    GetBiValue
    //    GetBiValue
    //    GetBiReg
    //    ApplyNewBi
    //    ApplyNewBiCWMinCheck (found in UpdateBiCWMinCheck)
    
    
    const double grd_Bi = static_cast<Derived*>(this)->GetBiGrad(i); // Gradient of Loss wrt to Bi
    
    this->Xtr[i] = std::abs(grd_Bi);  // Store absolute value of gradient for later steps
    
    const double old_Bi = this->B[i]; // copy of old Bi to adjust residuals if Bi updates
    
    const double nrb_Bi = static_cast<Derived*>(this)->GetBiValue(old_Bi, grd_Bi); 
    // Update Step for New No regularization No Bounds Bi:
    //                     n  r                 b     _Bi => nrb_Bi
    // Example
    // For CDL0: the update step is nrb_Bi = old_Bi + grd_Bi
    
    const double reg_Bi = static_cast<Derived*>(this)->GetBiReg(nrb_Bi); 
    // Ideal Bi with L1 and L2 regularization (no bounds)
    // Does not account for L0 regularziaton 
    // Example
    // For CDL0: reg_Bi = nrb_Bi as there is no L1, L2 parameters
    
    const double bnd_Bi = clamp(std::copysign(reg_Bi, nrb_Bi),
                                this->Lows[i], this->Highs[i]); 
    // Ideal Bi with regularization and bounds
    
    if (i < this->NoSelectK){
        // L0 penalty is not applied for NoSelectK Variables.
        // Only L1 and L2 (if either are used)
        if (std::abs(nrb_Bi) > this->lambda1){
            static_cast<Derived*>(this)->ApplyNewBi(i, old_Bi, bnd_Bi);
        } else if (old_Bi != 0) {
            static_cast<Derived*>(this)->ApplyNewBi(i, old_Bi, 0);
        }
    } else if (reg_Bi < this->thr){
        // If ideal non-bounded reg_Bi is less than threshold, coefficient is not worth setting.
        if (old_Bi != 0){
            static_cast<Derived*>(this)->ApplyNewBi(i, old_Bi, 0);
        }
    } else { 
        // Thus reg_Bi >= this->thr 
        
        const double delta_tmp = std::sqrt(reg_Bi*reg_Bi - this->thr2);
        // Due to numerical precision delta_tmp might be nan
        // Turns nans to 0.
        const double delta = (delta_tmp == delta_tmp) ? delta_tmp : 0;
        
        const double range_Bi = std::copysign(reg_Bi, nrb_Bi);
        
        
        if ((range_Bi - delta < bnd_Bi) && (bnd_Bi < range_Bi + delta)){
            // bnd_Bi exists in [bnd_Bi - delta, bnd_Bi + delta]
            // Therefore accept bnd_Bi
            static_cast<Derived*>(this)->ApplyNewBi(i, old_Bi, bnd_Bi);
        } else if (old_Bi != 0) {
            // Otherwise, reject bnd_Bi
            static_cast<Derived*>(this)->ApplyNewBi(i, old_Bi, 0);
        }
    }
}

template<class T, class Derived>
void CD<T, Derived>::UpdateBi(const std::size_t i){
    // Update a single coefficient of B for various CD Settings
    // The following functions are virtual and must be defined for any CD implementation.
    //    GetBiValue
    //    GetBiValue
    //    GetBiReg
    //    ApplyNewBi
    //    ApplyNewBiCWMinCheck (found in UpdateBiCWMinCheck)
    
    const double grd_Bi = static_cast<Derived*>(this)->GetBiGrad(i); // Gradient of Loss wrt to Bi

    //LOG("UpdateBi(" << i << ") Grad : " << grd_Bi << "\n");
        
    this->Xtr[i] = std::abs(grd_Bi);  // Store absolute value of gradient for later steps

    const double old_Bi = this->B[i]; // copy of old Bi to adjust residuals if Bi updates
    
    const double nrb_Bi = static_cast<Derived*>(this)->GetBiValue(old_Bi, grd_Bi); 
    // Update Step for New No regularization No Bounds Bi:
    //                     n  r                 b     _Bi => nrb_Bi
    // Example
    // For CDL0: the update step is nrb_Bi = old_Bi + grd_Bi
    
    const double reg_Bi = static_cast<Derived*>(this)->GetBiReg(nrb_Bi); 
    // Ideal Bi with L1 and L2 regularization (no bounds)
    // Does not account for L0 regularization 
    // Example
    // For CDL0: reg_Bi = nrb_Bi as there is no L1, L2 parameters
    
    const double new_Bi = std::copysign(reg_Bi, nrb_Bi); 
    
    if (i < this->NoSelectK){
        // L0 penalty is not applied for NoSelectK Variables.
        // Only L1 and L2 (if either are used)
        if (std::abs(nrb_Bi) > this->lambda1){
            static_cast<Derived*>(this)->ApplyNewBi(i, old_Bi, new_Bi);
        } else if (old_Bi != 0) {
            static_cast<Derived*>(this)->ApplyNewBi(i, old_Bi, 0);
        }
    } else if (reg_Bi < this->thr + lambda1_fudge_factor){
        // If ideal non-bounded reg_Bi is less than threshold, coefficient is not worth setting.
        if (old_Bi != 0){
            static_cast<Derived*>(this)->ApplyNewBi(i, old_Bi, 0);
        }
    } else { 
        static_cast<Derived*>(this)->ApplyNewBi(i, old_Bi, new_Bi);
    }
}

template<class T, class Derived>
bool CD<T, Derived>::UpdateBiCWMinCheck(const std::size_t i, const bool Cwmin){
    // See CD<T>::UpdateBi for documentation
    const double grd_Bi = static_cast<Derived*>(this)->GetBiGrad(i); 
    const double old_Bi = 0;
    
    this->Xtr[i] = std::abs(grd_Bi);  
    
    const double nrb_Bi = static_cast<Derived*>(this)->GetBiValue(old_Bi, grd_Bi); 
    const double reg_Bi = static_cast<Derived*>(this)->GetBiReg(nrb_Bi);
    const double new_Bi = std::copysign(reg_Bi, nrb_Bi);
    
    if (reg_Bi < this->thr + lambda1_fudge_factor){
        return Cwmin;
    } else {
        static_cast<Derived*>(this)->ApplyNewBiCWMinCheck(i, old_Bi, new_Bi);
        return false;
    }
}

template<class T, class Derived>
bool CD<T, Derived>::UpdateBiCWMinCheckWithBounds(const std::size_t i, const bool Cwmin){
    // See CD<T>::UpdateBi for documentation
    const double grd_Bi = static_cast<Derived*>(this)->GetBiGrad(i); 
    const double old_Bi = 0;
    
    this->Xtr[i] = std::abs(grd_Bi);  
    
    const double nrb_Bi = static_cast<Derived*>(this)->GetBiValue(old_Bi, grd_Bi); 
    const double reg_Bi = static_cast<Derived*>(this)->GetBiReg(nrb_Bi); 
    const double bnd_Bi = clamp(std::copysign(reg_Bi, nrb_Bi),
                                this->Lows[i], this->Highs[i]); 
    
    if (reg_Bi < this->thr){
        return Cwmin;
    } else {
        
        const double delta_tmp = std::sqrt(reg_Bi*reg_Bi - this->thr2);
        const double delta = (delta_tmp == delta_tmp) ? delta_tmp : 0;
        
        const double range_Bi = std::copysign(reg_Bi, nrb_Bi);
        if ((range_Bi - delta < bnd_Bi) && (bnd_Bi < range_Bi + delta)){
            static_cast<Derived*>(this)->ApplyNewBiCWMinCheck(i, old_Bi, bnd_Bi);
            return false;
        } else {
            return Cwmin;
        }
    }
}

/*
 * 
 *  CDBase
 * 
 */

template<class T>
CDBase<T>::CDBase(const T& Xi, const Eigen::ArrayXd& yi, const Params<T>& P) :
    ModelParams{P.ModelParams}, CyclingOrder{P.CyclingOrder}, MaxIters{P.MaxIters},
    rtol{P.rtol}, atol{P.atol}, ActiveSet{P.ActiveSet}, y{yi}, 
    ActiveSetNum{P.ActiveSetNum}, Lows{P.Lows}, Highs{P.Highs}
{
        
    this->lambda0 = P.ModelParams[0];
    this->lambda1 = P.ModelParams[1];
    this->lambda2 = P.ModelParams[2];
    
    this->result.ModelParams = P.ModelParams; 
    this->NoSelectK = P.NoSelectK;
    
    this->Xtr = P.Xtr; 
    this->Iter = P.Iter;
    
    this->isSparse = std::is_same<T,Eigen::SparseMatrix<double>>::value;
    
    this->b0 = P.b0;
    this->intercept = P.intercept;
    
    this->withBounds = P.withBounds;
    
    this->X = &Xi;
    
    this->n = X->rows();
    this->p = X->cols();
    
    if (P.Init == 'u') {
        this->B = P.InitialSol;
    } else {
        this->B = beta_vector::Zero(p);
    }
    
    if (CyclingOrder == 'u') {
        this->Order = P.Uorder;
    } else if (CyclingOrder == 'c') {
        std::vector<std::size_t> cyclic(p);
        std::iota(std::begin(cyclic), std::end(cyclic), 0);
        this->Order = cyclic;
    }
    
    this->CurrentIters = 0;
}

template<class T>
FitResult<T> CDBase<T>::Fit(){
    if (this->withBounds){
        return this->_FitWithBounds();
    } else{
        return this->_Fit();
    }
}

template class CDBase<Eigen::MatrixXd>;
template class CDBase<Eigen::SparseMatrix<double>>;

/*
 * 
 *  CD
 * 
 */

template<class T, class Derived>
void CD<T, Derived>::UpdateSparse_b0(Eigen::ArrayXd& r){
    // Only run for regression when T is arma::sp_mat and intercept is True.
    // r is this->r on outer scope;                                                           
    const double new_b0 = r.mean();
    r -= new_b0;
    this->b0 += new_b0;
}


template<class T, class Derived>
bool CD<T, Derived>::isConverged() {
    LOG("isConverged Start");
    this->CurrentIters += 1; // keeps track of the number of calls to Converged
    const double objectiveold = this->objective;
    this->objective = this->Objective();
    LOG("Objective() END");
    return ((std::abs(objectiveold - this->objective) <= this->rtol*objectiveold) 
                || (this->objective <= this->atol)); 
}

template<class T, class Derived>
void CD<T, Derived>::RestrictSupport() {
    LOG("RestrictSupport Start");
    if (has_same_support(this->B, this->Bprev)) {
        this->SameSuppCounter += 1;
        
        if (this->SameSuppCounter == this->ActiveSetNum - 1) {
            std::vector<std::size_t> NewOrder = nnzIndicies(this->B);
            
            std::sort(NewOrder.begin(),
                      NewOrder.end(), 
                      [this](std::size_t i, std::size_t j) {return this->Order[i] < this->Order[j] ;});
            
            this->OldOrder = this->Order;
            this->Order = NewOrder;
            this->ActiveSet = false;
            this->Stabilized = true;
            
        }
        
    } else {
        this->SameSuppCounter = 0;
    }
    LOG("RestrictSupport END");
    
}

template<class T, class Derived>
bool CD<T, Derived>::CWMinCheckWithBounds() {
    LOG("CWMinCheckWithBounds Start");
    std::vector<std::size_t> S = nnzIndicies(this->B);
    
    std::vector<std::size_t> Sc;
    set_difference(
        this->Range1p.begin(),
        this->Range1p.end(),
        S.begin(),
        S.end(),
        back_inserter(Sc));
    
    bool Cwmin = true;
    for (auto& i : Sc) {
        Cwmin = this->UpdateBiCWMinCheckWithBounds(i, Cwmin);
    }
    LOG("CWMinCheckWithBounds End");
    return Cwmin;
}

template<class T, class Derived>
bool CD<T, Derived>::CWMinCheck() {
    LOG("CWMinCheck Start");
    std::vector<std::size_t> S = nnzIndicies(this->B);
    
    std::vector<std::size_t> Sc;
    set_difference(
        this->Range1p.begin(),
        this->Range1p.end(),
        S.begin(),
        S.end(),
        back_inserter(Sc));
    
    bool Cwmin = true;
    for (auto& i : Sc) {
        Cwmin = this->UpdateBiCWMinCheck(i, Cwmin);
    }
    LOG("CWMinCheck END");
    
    return Cwmin;
}


template<class T, class Derived>
CD<T, Derived>::CD(const T& Xi, const Eigen::ArrayXd& yi, const Params<T>& P) : CDBase<T>(Xi, yi, P){
    Range1p.resize(this->p);
    std::iota(std::begin(Range1p), std::end(Range1p), 0);
    ScreenSize = P.ScreenSize;
}

// template class CD<arma::mat>;
// template class CD<arma::sp_mat>;

/*
 * 
 *  CDSwaps
 * 
 */

template <class T>
CDSwaps<T>::CDSwaps(const T& Xi, const Eigen::ArrayXd& yi, const Params<T>& Pi) : CDBase<T>(Xi, yi, Pi){
    MaxNumSwaps = Pi.MaxNumSwaps;
    P = Pi;
}

template class CDSwaps<Eigen::MatrixXd>;
template class CDSwaps<Eigen::SparseMatrix<double>>;

#endif
