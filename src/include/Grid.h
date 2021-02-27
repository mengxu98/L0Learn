#ifndef GRID_H
#define GRID_H
#include <tuple>
#include <set>
#include <memory>
#include "RcppEigen.h"
#include "GridParams.h"
#include "FitResult.h"
#include "Grid1D.h"
#include "Grid2D.h"
#include "Normalize.h"
#include "logging.h"


template <class T>
class Grid {
    private:
        T Xscaled;
        Eigen::ArrayXd yscaled;
        Eigen::ArrayXd BetaMultiplier;
        Eigen::ArrayXd meanX;
        double meany;
        double scaley;

    public:

        GridParams<T> PG;

        std::vector< std::vector<double> > Lambda0;
        std::vector<double> Lambda12;
        std::vector< std::vector<std::size_t> > NnzCount;
        std::vector< std::vector<Eigen::SparseVector<double>>> Solutions;
        std::vector< std::vector<double> >Intercepts;
        std::vector< std::vector<bool> > Converged;

        Grid(const T& X, const Eigen::ArrayXd& y, const GridParams<T>& PG);
        //~Grid();

        void Fit();

};

#endif
