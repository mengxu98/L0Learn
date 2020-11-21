#include "CD.h"

template <class T>
CDBase<T>::CDBase(const T& Xi, const Eigen::VectorXd& yi, const Params<T>& P) :
    ModelParams{P.ModelParams}, CyclingOrder{P.CyclingOrder}, MaxIters{P.MaxIters},
    Tol{P.Tol}, ActiveSet{P.ActiveSet}, ActiveSetNum{P.ActiveSetNum} 
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
    this->X = &Xi;
    this->y = &yi;
    
    this->n = X->rows();
    this->p = X->cols();
    
    if (P.Init == 'u') {
        this->B = *(P.InitialSol);
    } else if (P.Init == 'r') {
        // Random values from [-1, 1]
        Eigen::RowVectorXd row_indices_d = Eigen::RowVectorXd::Random(P.RandomStartSize);
        row_indices_d = row_indices_d.array() + 1.0;
        row_indices_d = row_indices_d.array() * (p-1)/2; // Random values from [0, p-1]
        
        Eigen::RowVectorXi row_indices = row_indices_d.cast <std::size_t> ();
        // Random Integers values from {0, 1, 2, ..., p-1}
      
        std::vector<std::size_t> unique_row_indices(P.RandomStartSize);
        for (auto i = 0; i < p-1; ++i){
          unique_row_indices.push_back(row_indices(i));
        }
        std::sort(unique_row_indices.begin(), unique_row_indices.end());
        std::unique(unique_row_indices.begin(), unique_row_indices.end());
        // Sorted Unique List of Indicies
        
        std::vector<Eigen::Triplet<double>> tripletList;
        
        auto nnz = unique_row_indices.size();
        tripletList.reserve(nnz);
        
        Eigen::RowVectorXd random_values = Eigen::VectorXd::Random(nnz);
        Eigen::SparseVector<double> B(p);
        for (auto i=0; i < nnz; ++i){
            B.coeffRef(i) = random_values(i);
        }
        this->B = B;
    } else {
        this->B = Eigen::SparseVector<double>(p); // Initialized to zeros
    }
    
    if (CyclingOrder == 'u') {
        this->Order = P.Uorder;
    } else if (CyclingOrder == 'c') {
        std::vector<std::size_t> cyclic(p);
        std::iota(std::begin(cyclic), std::end(cyclic), 0);
        this->Order = cyclic;
    }
    
    this->Lows = P.Lows;
    this->Highs = P.Highs;
 
    this->CurrentIters = 0;
}

template <class T>
void CD<T>::UpdateSparse_b0(Eigen::VectorXd& r){
    // Only run for regression when T is Eigen::SparseMatrix<double> and intercept is True.
    // r is this->r on outer scope;                                                           
    const double new_b0 = r.mean();
    r = r.array() - new_b0;
    this->b0 += new_b0;
}


template <class T>
void CD<T>::UpdateBi(const std::size_t i){
    // Update a single coefficient of B for various CD Settings
    // The following functions are virtual and must be defined for any CD implementation.
    //    GetBiValue
    //    GetBiValue
    //    GetBiReg
    //    ApplyNewBi
    //    ApplyNewBiCWMinCheck (found in UpdateBiCWMinCheck)
    
    
    const double grd_Bi = this->GetBiGrad(i); // Gradient of Loss wrt to Bi
  
    (*this->Xtr)[i] = std::abs(grd_Bi);  // Store absolute value of gradient for later steps
    
    const double old_Bi = this->B.coeffRef(i); // copy of old Bi to adjust residuals if Bi updates
    
    const double nrb_Bi = this->GetBiValue(old_Bi, grd_Bi); 
    // Update Step for New No regularization No Bounds Bi:
    //                     n  r                 b     _Bi => nrb_Bi
    // Example
    // For CDL0: the update step is nrb_Bi = old_Bi + grd_Bi
    
    const double reg_Bi = this->GetBiReg(nrb_Bi); 
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
          this->ApplyNewBi(i, old_Bi, bnd_Bi);
        } else {
          this->ApplyNewBi(i, old_Bi, 0);
        }
    } else if (reg_Bi < this->thr){
        // If ideal non-bounded reg_Bi is less than threshold, coefficient is not worth setting.
        if (old_Bi != 0){
            this->ApplyNewBi(i, old_Bi, 0);
        }
    } else { 
      // Thus reg_Bi >= this->thr 
      
      const double delta_tmp = std::sqrt(reg_Bi*reg_Bi - this->thr2);
      // Due to numerical precisions delta_tmp might be nan/
      const double delta = (delta_tmp == delta_tmp) ? delta_tmp : 0;
      // Turns nans to 0.
      
      const double range_Bi = std::copysign(reg_Bi, nrb_Bi);
      
      
      if ((range_Bi - delta < bnd_Bi) && (bnd_Bi < range_Bi + delta)){
          // bnd_Bi exists in [bnd_Bi - delta, bnd_Bi + delta]
          // Therefore accept bnd_Bi
          this->ApplyNewBi(i, old_Bi, bnd_Bi);
      } else {
          // Otherwise, reject bnd_Bi
          this->ApplyNewBi(i, old_Bi, 0);
      }
    }
}

template <class T>
bool CD<T>::UpdateBiCWMinCheck(const std::size_t i, const bool Cwmin){
  // See CD<T>::UpdateBi for documentation
  const double grd_Bi = this->GetBiGrad(i); 
  
  (*this->Xtr)[i] = std::abs(grd_Bi);  
  
  const double nrb_Bi = this->GetBiValue(0, grd_Bi); 
  const double reg_Bi = this->GetBiReg(nrb_Bi); 
  const double bnd_Bi = clamp(std::copysign(reg_Bi, nrb_Bi),
                              this->Lows[i], this->Highs[i]); 

  if (i < this->NoSelectK){
    if (std::abs(nrb_Bi) > this->lambda1){
      this->ApplyNewBiCWMinCheck(i, 0, bnd_Bi);
      return false;
    } else {
      return Cwmin;
    }
      
  } else if (reg_Bi < this->thr){
    return Cwmin;
  } else {
    
    const double delta_tmp = std::sqrt(reg_Bi*reg_Bi - this->thr2);
    const double delta = (delta_tmp == delta_tmp) ? delta_tmp : 0;

    const double range_Bi = std::copysign(reg_Bi, nrb_Bi);
    if ((range_Bi - delta < bnd_Bi) && (bnd_Bi < range_Bi + delta)){
      this->ApplyNewBiCWMinCheck(i, 0, bnd_Bi);
      return false;
    } else {
      return Cwmin;
    }
  }
}


template <class T>
bool CD<T>::Converged() {
    this->CurrentIters += 1; // keeps track of the number of calls to Converged
    double objectiveold = this->objective;
    this->objective = this->Objective();
    return std::abs(objectiveold - this->objective) <= this->Tol*objectiveold; 
}

template <class T>
void CD<T>::SupportStabilized() {
    
    bool SameSupp = true;
    
    if (this->Bprev.nonZeros() != this->B.nonZeros()) {
        SameSupp = false;
    } else {  // same number of nnz and Supp is sorted
      
      std::vector<std::size_t> B_indicies;
      std::vector<std::size_t> B_prev_indicies;
      
      for (Eigen::SparseVector<double>::InnerIterator it(this->B); it; ++it)
          B_indicies.push_back(it.index());
      
      for (Eigen::SparseVector<double>::InnerIterator it(this->Bprev); it; ++it)
          B_prev_indicies.push_back(it.index());
        
      std::sort(B_indicies.begin(), B_indicies.end());
      std::sort(B_prev_indicies.begin(), B_prev_indicies.end());
      
      SameSupp = B_indicies == B_prev_indicies;
    }
    
    if (SameSupp) {
        this->SameSuppCounter += 1;
        
        if (this->SameSuppCounter == this->ActiveSetNum - 1) {
            std::vector<std::size_t> NewOrder(this->B.nonZeros());
            
            for (Eigen::SparseVector<double>::InnerIterator it(this->B); it; ++it)
                NewOrder.push_back(it.index());
            
            std::sort(NewOrder.begin(), NewOrder.end(), [this](std::size_t i, std::size_t j) {return this->Order[i] <  this->Order[j] ;});
            
            this->OldOrder = this->Order;
            this->Order = NewOrder;
            this->ActiveSet = false;
            this->Stabilized = true;
            
        }
        
    } else {
        this->SameSuppCounter = 0;
    }
    
}

template class CDBase<Eigen::MatrixXd>;
template class CDBase<Eigen::SparseMatrix<double>>;


template <class T>
CD<T>::CD(const T& Xi, const Eigen::VectorXd& yi, const Params<T>& P) : CDBase<T>(Xi, yi, P){
    Range1p.resize(this->p);
    std::iota(std::begin(Range1p), std::end(Range1p), 0);
    ScreenSize = P.ScreenSize;
}

template class CD<Eigen::MatrixXd>;
template class CD<Eigen::SparseMatrix<double>>;


template <class T>
CDSwaps<T>::CDSwaps(const T& Xi, const Eigen::VectorXd& yi, const Params<T>& Pi) : CDBase<T>(Xi, yi, Pi){
    MaxNumSwaps = Pi.MaxNumSwaps;
    P = Pi;
}

template class CDSwaps<Eigen::MatrixXd>;
template class CDSwaps<Eigen::SparseMatrix<double>>;
