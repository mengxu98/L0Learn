#include "Grid1D.h"

#include <chrono>
#include <thread>


template <class T>
Grid1D<T>::Grid1D(const T& Xi, const Eigen::ArrayXd& yi, const GridParams<T>& PG) : y{yi}{
    // automatically selects lambda_0 (but assumes other lambdas are given in PG.P.ModelParams)
    
    LOG("Grid1D Constructor");
    X = &Xi;
    p = Xi.cols();
    LambdaMinFactor = PG.LambdaMinFactor;
    ScaleDownFactor = PG.ScaleDownFactor;
    P = PG.P;
    P.Xtr = std::vector<double>(Xi.cols());
    P.ytX = Eigen::RowVectorXd::Zero(Xi.cols());
    //P.D = std::map<std::size_t,Eigen::RowVectorXd>();
    P.r = Eigen::ArrayXd::Zero(Xi.rows());
    Xtr = P.Xtr;
    ytX = P.ytX;
    NoSelectK = P.NoSelectK;
    
    LambdaU = PG.LambdaU;
    
    if (!LambdaU) {
        G_ncols = PG.G_ncols;
    } else {
        G_ncols = PG.Lambdas.rows(); // override the user's ncols if LambdaU = 1
    }
    
    G.reserve(G_ncols);
    if (LambdaU) {
        Lambdas = PG.Lambdas;
    } // user-defined lambda0 grid
    /*
     else {
     Lambdas.reserve(G_ncols);
     Lambdas.push_back((0.5*arma::square(y->t() * *X)).max());
     }
     */
    NnzStopNum = PG.NnzStopNum;
    Refine = PG.Refine;
    PartialSort = PG.PartialSort;
    XtrAvailable = PG.XtrAvailable;
    if (XtrAvailable) {
        ytXmax2d = PG.ytXmax;
        Xtr = PG.Xtr;
    }
}

template <class T>
Grid1D<T>::~Grid1D() {
    // delete all dynamically allocated memory
    // delete P.Xtr;
    // delete P.ytX;
    // delete P.D;
    // delete P.r;
}


template <class T>
std::vector<std::unique_ptr<FitResult<T>>> Grid1D<T>::Fit() {

    LOG("Grid1D Fit");
    if (P.Specs.L0 || P.Specs.L0L2 || P.Specs.L0L1) {
        bool scaledown = false;
        
        double Lipconst;
        Eigen::ArrayXd Xtrarma = Eigen::ArrayXd::Zero(y.rows());
        if (P.Specs.Logistic) {
            if (!XtrAvailable) {
                Xtrarma = 0.5 * (y.transpose().matrix() * *X).array().abs().transpose();
            } // = gradient of logistic loss at zero}
            Lipconst = 0.25 + 2 * P.ModelParams[2];
        } else if (P.Specs.SquaredHinge) {
            if (!XtrAvailable) {
                // gradient of loss function at zero
                Xtrarma = 2 * (y.transpose().matrix() * *X).array().abs().transpose();
            } 
            Lipconst = 2 + 2 * P.ModelParams[2];
        } else {
            if (!XtrAvailable) {
                ytX = y.transpose().matrix() * *X;
                Xtrarma = ytX.array().abs().transpose(); // Least squares
            }
            Lipconst = 1 + 2 * P.ModelParams[2];
            P.r = y - P.b0; // B = 0 initially
        }
        
        double ytXmax;
        if (!XtrAvailable) {
            Xtr = std::vector<double>(Xtrarma.data(), Xtrarma.data() + Xtrarma.rows() * Xtrarma.cols());
            ytXmax = Xtrarma.maxCoeff();
        } else {
            ytXmax = ytXmax2d;
        }
        
        double lambdamax = ((ytXmax - P.ModelParams[1]) * (ytXmax - P.ModelParams[1])) / (2 * (Lipconst));
        
        // Rcpp::Rcout << "lambdamax: " << lambdamax << "\n";
        
        if (!LambdaU) {
            P.ModelParams[0] = lambdamax;
        } else {
            P.ModelParams[0] = Lambdas[0];
        }
        
        // Rcpp::Rcout << "P ModelParams: {" << P.ModelParams[0] << ", " << P.ModelParams[1] << ", " << P.ModelParams[2] << ", " << P.ModelParams[3] <<   "}\n";
        
        P.Init = 'z';
        
        
        //std::cout<< "Lambda max: "<< lambdamax << std::endl;
        //double lambdamin = lambdamax*LambdaMinFactor;
        //Lambdas = arma::logspace(std::log10(lambdamin), std::log10(lambdamax), G_ncols);
        //Lambdas = arma::flipud(Lambdas);
        
        
        //std::size_t StopNum = (X->n_rows < NnzStopNum) ? X->n_rows : NnzStopNum;
        std::size_t StopNum = NnzStopNum;
        //std::vector<double>* Xtr = P.Xtr;
        std::vector<std::size_t> idx(p);
        double Xrmax;
        bool prevskip = false; //previous grid point was skipped
        bool currentskip = false; // current grid point should be skipped
        
        for (std::size_t i = 0; i < G_ncols; ++i) {
            Rcpp::checkUserInterrupt();
            // Rcpp::Rcout << "Grid1D: " << i << "\n";
            FitResult<T> *  prevresult = new FitResult<T>; // prevresult is ptr to the prev result object
            //std::unique_ptr<FitResult> prevresult;
            if (i > 0) {
                //prevresult = std::move(G.back());
                *prevresult = *(G.back());
                
            }
      
            currentskip = false;
            
            if (!prevskip) {
                
                std::iota(idx.begin(), idx.end(), 0); // make global class var later
                // Exclude the first NoSelectK features from sorting.
                if (PartialSort && p > 5000 + NoSelectK)
                    std::partial_sort(idx.begin() + NoSelectK, idx.begin() + 5000 + NoSelectK, idx.end(), [this](std::size_t i1, std::size_t i2) {return Xtr[i1] > Xtr[i2] ;});
                else
                    std::sort(idx.begin() + NoSelectK, idx.end(), [this](std::size_t i1, std::size_t i2) {return Xtr[i1] > Xtr[i2] ;});
                P.CyclingOrder = 'u';
                P.Uorder = idx; // can be made faster
                
                //
                Xrmax = Xtr[idx[NoSelectK]];
                
                if (i > 0) {
                    std::vector<std::size_t> Sp = nnzIndicies(prevresult->B);
                    
                    for(std::size_t l = NoSelectK; l < p; ++l) {
                        if ( std::binary_search(Sp.begin(), Sp.end(), idx[l]) == false ) {
                            Xrmax = Xtr[idx[l]];
                            //Rcpp::Rcout << "l:" << l << "\n";
                            //std::cout<<"Grid Iteration: "<<i<<" Xrmax= "<<Xrmax<<std::endl;
                            break;
                        }
                    }
                }
            }
            
            // Following part assumes that lambda_0 has been set to the new value
            if(i >= 1 && !scaledown && !LambdaU) {
                P.ModelParams[0] = (((Xrmax - P.ModelParams[1]) * (Xrmax - P.ModelParams[1])) / (2 * (Lipconst))) * 0.99; // for numerical stability issues.
                
                if (P.ModelParams[0] >= prevresult->ModelParams[0]) {
                    P.ModelParams[0] = prevresult->ModelParams[0] * 0.97;
                } // handles numerical instability.
            } else if (i >= 1 && !LambdaU) {
                P.ModelParams[0] = std::min(P.ModelParams[0] * ScaleDownFactor, (((Xrmax - P.ModelParams[1]) * (Xrmax - P.ModelParams[1])) / (2 * (Lipconst))) * 0.97 );
                // add 0.9 as an R param
            } else if (i >= 1 && LambdaU) {
                P.ModelParams[0] = Lambdas[i];
            }
            
            //Rcpp::Rcout << "Xrmax: " << Xrmax << "\n";
            //Rcpp::Rcout << "P.ModelParams[0]: " << P.ModelParams[0] << "\n";
           
            if (!currentskip) {
            
                auto Model = make_CD(*X, y, P);
              
                //Rcpp::Rcout << "Grid1D: Model->Fit"<< print_i++ << "\n";
                //std::this_thread::sleep_for(std::chrono::milliseconds(100));
                
                std::unique_ptr<FitResult<T>> result(new FitResult<T>);
                LOG("Model Fit start");
                *result = Model->Fit();
                LOG("Model Fit end");
                //Rcpp::Rcout << "Grid1D: Model->Fit DONE"<< print_i++ << "\n";
                //std::this_thread::sleep_for(std::chrono::milliseconds(100));
                delete Model;
                
                //Rcpp::Rcout << "Grid1D:"<< print_i++ << "\n";
                //std::this_thread::sleep_for(std::chrono::milliseconds(100));
                scaledown = false;
                if (i >= 1) {
                    //Rcpp::Rcout << "Grid1D:"<< print_i++ << "\n";
                    //std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    std::vector<std::size_t> Spold = nnzIndicies(prevresult->B);
                    //Rcpp::Rcout << "Grid1D:"<< print_i++ << "\n";
                    //std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    std::vector<std::size_t> Spnew = nnzIndicies(result->B);
                   
                    bool samesupp = false;
                    //Rcpp::Rcout << "|Spold|: " << Spold.size() << ", |Spnew|: " << Spnew.size() <<" \n";
                    
                    if (Spold == Spnew) {
                        samesupp = true;
                        scaledown = true;
                    }
                    
                    // //
                    // 
                    // if (samesupp) {
                    //     scaledown = true;
                    // } // got same solution
                }
                
                //Rcpp::Rcout << "Grid1D: Model Spold"<< print_i++ << "\n";
                //std::this_thread::sleep_for(std::chrono::milliseconds(100));
                
                //else {scaledown = false;}
                G.push_back(std::move(result));
                
                
                if(n_nonzero(G.back()->B) >= StopNum) {
                    break;
                }
                //result->B.t().print();
                P.InitialSol = G.back()->B;
                P.b0 = G.back()->b0;
                P.r = G.back()->r;
                
                // Udate: After 1.1.0, P.r is automatically updated by the previous call to CD
                
                //Rcpp::Rcout << "Grid1D:  G.back()->B;"<< print_i++ << "\n";
                //std::this_thread::sleep_for(std::chrono::milliseconds(100));
                
            }
            
            delete prevresult;
            
            
            P.Init = 'u';
            P.Iter += 1;
            prevskip = currentskip;
        }
    }
    
    return std::move(G);
}


template class Grid1D<Eigen::MatrixXd>;
template class Grid1D<Eigen::SparseMatrix<double>>;
