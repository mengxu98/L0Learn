#ifndef GRID_H
#define GRID_H
#include <tuple>
#include <set>
#include <memory>
#include <RcppEigen.h>
#include "GridParams.h"
#include "FitResult.h"
#include "Grid1D.h"
#include "Grid2D.h"
#include "Normalize.h"

template <class T>
class Grid {
    private:
        T Xscaled;
       Eigen::VectorXd yscaled;
       Eigen::VectorXd BetaMultiplier;
       Eigen::VectorXd meanX;
        double meany;
        double scaley;

    public:

        GridParams<T> PG;

        std::vector< std::vector<double> > Lambda0;
        std::vector<double> Lambda12;
        std::vector< std::vector<std::size_t> > NnzCount;
        std::vector< std::vector<Eigen::SparseMatrix<double>> > Solutions;
        std::vector< std::vector<double> >Intercepts;
        std::vector< std::vector<bool> > Converged;

        Grid(const T& X, constEigen::VectorXd& y, const GridParams<T>& PG);
        //~Grid();

        void Fit();

};

#endif
