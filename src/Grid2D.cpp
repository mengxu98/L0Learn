#include "Grid2D.h"

template <class T>
Grid2D<T>::Grid2D(const T& Xi, const Eigen::ArrayXd& yi, const GridParams<T>& PGi) : y{yi}
{
    // automatically selects lambda_0 (but assumes other lambdas are given in PG.P.ModelParams)
    X = &Xi;
    p = Xi.cols();
    PG = PGi;
    G_nrows = PG.G_nrows;
    G_ncols = PG.G_ncols;
    G.reserve(G_nrows);
    Lambda2Max = PG.Lambda2Max;
    Lambda2Min = PG.Lambda2Min;
    LambdaMinFactor = PG.LambdaMinFactor;
    
    P = PG.P;
}

template <class T>
Grid2D<T>::~Grid2D(){
    //delete Xtr;
    if (PG.P.Specs.Logistic)
        delete PG.P.Xy;
    if (PG.P.Specs.SquaredHinge)
        delete PG.P.Xy;
}

template <class T>
std::vector< std::vector<std::unique_ptr<FitResult<T>> > > Grid2D<T>::Fit() {
    Eigen::ArrayXd Xtrarma;
    
    if (PG.P.Specs.Logistic) {
        auto n = X->rows();
        double b0 = 0;
        Eigen::ArrayXd ExpyXB =  Eigen::ArrayXd::Ones(n);
        if (PG.intercept) {
            for (std::size_t t = 0; t < 50; ++t) {
                double partial_b0 = - ( y / (1 + ExpyXB) ).sum();
                b0 -= partial_b0 / (n * 0.25); // intercept is not regularized
                ExpyXB = (b0*y).exp();
            }
        }
        PG.P.b0 = b0;
        Xtrarma = (- (y /(1+ExpyXB)).transpose().matrix() * *X).array().abs().transpose(); // = gradient of logistic loss at zero
        //Xtrarma = 0.5 * arma::abs(y->t() * *X).t(); // = gradient of logistic loss at zero
        
        T Xy =  (*X).cwiseProduct(y.matrix()); // X->each_col() % *y;
        PG.P.Xy = new T;
        *PG.P.Xy = Xy;
    }
    
    else if (PG.P.Specs.SquaredHinge) {
        auto n = X->rows();
        double b0 = 0;
        Eigen::ArrayXd onemyxb =  Eigen::ArrayXd::Ones(n);
        if (PG.intercept){
            for (std::size_t t = 0; t < 50; ++t){
                double partial_b0 = 2*(onemyxb.max(0) * -y).sum();
                b0 -= partial_b0 / (n * 2); // intercept is not regularized
                onemyxb = 1 - (y * b0);
            }
        }
        PG.P.b0 = b0;
        Xtrarma = 2 * ((y * onemyxb.max(0)).transpose().matrix()* *X).array().abs().transpose(); // = gradient of loss function at zero
        //Xtrarma = 2 * arma::abs(y->t() * *X).t(); // = gradient of loss function at zero
        T Xy = (*X).cwiseProduct(y.matrix()); // X->each_col() % *y;
        PG.P.Xy = new T;
        *PG.P.Xy = Xy;
        
    } else {
        Xtrarma = (y.transpose().matrix() * *X).array().abs().transpose();
    }
    
    
    double ytXmax = Xtrarma.maxCoeff();
    
    std::size_t index;
    if (PG.P.Specs.L0L1) {
        index = 1;
        if(G_nrows != 1) {
            Lambda2Max = ytXmax;
            Lambda2Min = Lambda2Max * LambdaMinFactor;
        }
    } else if (PG.P.Specs.L0L2) {
        index = 2;
    }
    Eigen::ArrayXd Lambdas2 = Eigen::pow(10, Eigen::ArrayXd::LinSpaced(Lambda2Min, Lambda2Max, G_nrows));
    //Eigen::ArrayXd Lambdas2 = arma::logspace(std::log10(Lambda2Min), std::log10(Lambda2Max), G_nrows);
    Lambdas2 = Lambdas2.reverse();
    
    std::vector<double> Xtrvec = std::vector<double>(Xtrarma.data(), 
                                                     Xtrarma.data() + Xtrarma.rows() * Xtrarma.cols());
    
    
    Xtr = std::vector<double>(X->cols()); // needed! careful
    
    
    PG.XtrAvailable = true;
    // Rcpp::Rcout << "Grid2D Start\n";
    for(std::size_t i=0; i<Lambdas2.size();++i) { //auto &l : Lambdas2
        // Rcpp::Rcout << "Grid1D Start: " << i << "\n";
        Xtr = Xtrvec;
        
        PG.Xtr = Xtr;
        PG.ytXmax = ytXmax;
        
        PG.P.ModelParams[index] = Lambdas2[i];
        
        if (PG.LambdaU)
            PG.Lambdas = Eigen::VectorXd::Map(PG.LambdasGrid[i].data(), PG.LambdasGrid[i].size());
        
        //std::vector<std::unique_ptr<FitResult>> Gl();
        //auto Gl = Grid1D(*X, *y, PG).Fit();
        // Rcpp::Rcout << "Grid1D Start: " << i << "\n";
        G.push_back(std::move(Grid1D<T>(*X, y, PG).Fit()));
    }
    
    return std::move(G);
    
}

template class Grid2D<Eigen::MatrixXd>;
template class Grid2D<Eigen::SparseMatrix<double>>;
