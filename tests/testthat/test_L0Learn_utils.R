library("Matrix")
library("testthat")
library("L0Learn")
library("tilting")

n = 50
p = 100
k = 10
tmp <-  L0Learn::GenSynthetic(n=n, p=p, k=2, seed=1, rho=1)
X <- tmp[[1]] + rnorm(p)
y <- tmp[[2]]
tol = 1e-4

if (sum(apply(X, 2, sd) == 0)) {
    stop("X needs to have non-zero std for each column")
}

X_sparse <- as(X, "dgCMatrix")

test_that('matrix_column_get dense', {
    x1 <- as.matrix(X[, 1])
    x2 <- .Call('_L0Learn_R_matrix_column_get_dense', X, 0) # C++ and R use different indexes
    expect_equal(as.vector(x1), as.vector(x2))
})

test_that('matrix_column_get sparse', {
    x1 <- as.matrix(X_sparse[, 1])
    x2 <- .Call('_L0Learn_R_matrix_column_get_sparse', X_sparse, 0) # C++ and R use different indexes
    expect_equal(as.vector(x1), as.vector(x2))
})

test_that('matrix_rows_get dense', {
    x1 <- X[1:4, ]
    x2 <- .Call("_L0Learn_R_matrix_rows_get_dense", X, 0:3)
    expect_equal(x1, x2)
    
    test_rows = c(1, 4, 10, 5, 5, 10, n-1)
    x1 <- X[test_rows, ]
    x2 <- .Call("_L0Learn_R_matrix_rows_get_dense", X, test_rows-1)
    expect_equal(x1, x2)
})

test_that('matrix_rows_get sparse', {
    x1 <- X_sparse[1:4, ]
    x2 <- .Call("_L0Learn_R_matrix_rows_get_sparse", X_sparse, 0:3)
    expect_equal(x1, x2)
    
    rows = 1:n
    test_rows = sample(rows, floor(rows/4))
    x1 <- X_sparse[test_rows, ]
    x2 <- .Call("_L0Learn_R_matrix_rows_get_sparse", X_sparse, test_rows-1)
    expect_equal(x1, x2)
})

# test_that('matrix_vector_schur_produce dense', {
#     x1 <- X*as.vector(y)
#     x2 <- .Call('_L0Learn_R_matrix_vector_schur_product_dense', X, y)
#     expect_equal(x1, x2)
# })
# 
# test_that('matrix_vector_schur_produce sparse', {
#     x1 <- X_sparse*as.vector(y)
#     x2 <- .Call('_L0Learn_R_matrix_vector_schur_product_sparse', X_sparse, y)
#     expect_equal(x1, x2)
# })


# test_that('matrix_vector_divide dense', {
#     x1 <- X/as.vector(y)
#     x2 <- .Call('_L0Learn_R_matrix_vector_divide_dense', X, y)
#     expect_equal(x1, x2)
# })
# 
# test_that('matrix_vector_divide sparse', {
#     x1 <- X_sparse/as.vector(y)
#     x2 <- .Call('_L0Learn_R_matrix_vector_divide_sparse', X_sparse, y)
#     expect_equal(x1, x2)
# })

test_that('matrix_column_sums dense', {
    x1 <- colSums(X)
    x2 <- as.vector(.Call('_L0Learn_R_matrix_column_sums_dense', X))
    expect_equal(x1, x2)
})

test_that('matrix_column_sums sparse', {
    x1 <- colSums(X_sparse)
    x2 <- as.vector(.Call('_L0Learn_R_matrix_column_sums_sparse', X_sparse))
    expect_equal(x1, x2)
})

test_that('matrix_column_dot dense', {
    x1 <- X[,1]%*%y
    x2 <- .Call('_L0Learn_R_matrix_column_dot_dense', X, 0, y)
    expect_equal(as.double(x1), as.double(x2))
})

test_that('matrix_column_dot sparse', {
    x1 <- X_sparse[,1]%*%y
    x2 <- .Call('_L0Learn_R_matrix_column_dot_sparse', X_sparse, 0, y)
    expect_equal(as.double(x1), as.double(x2))
})

# test_that('matrix_column_mult dense', {
#     c = 3.14
#     x1 <- X[,1]*c
#     x2 <- .Call('_L0Learn_R_matrix_column_mult_dense', X, 0, c)
#     expect_equal(as.double(x1), as.double(x2))
# })
# 
# test_that('matrix_column_mult sparse', {
#     c = 3.14
#     x1 <- X_sparse[,1]*c
#     x2 <- .Call('_L0Learn_R_matrix_column_mult_sparse', X_sparse, 0, c)
#     expect_equal(as.double(x1), as.double(x2))
# })

center_colmeans <- function(x) {
    xcenter = colMeans(x)
    x - rep(xcenter, rep.int(nrow(x), ncol(x)))
}

test_that('matrix_normalize dense', {
    for (norm in c(TRUE, FALSE)){
        if (norm){
            X_norm <- center_colmeans(X)
            expect_equal(colMeans(X_norm), 0*colMeans(X_norm))
        } else {
            X_norm <- as.matrix(X)
        }
        X_norm_copy = as.matrix(X_norm)
        expect_equal(X_norm, X_norm_copy)
        
        x1 <- .Call("_L0Learn_R_matrix_normalize_dense", X_norm)
        
        expect_equal(X_norm, X_norm_copy) # R should not modify X_norm
        
        expect_equal(col.norm(X_norm), as.vector(x1$ScaleX))
        expect_equal(X_norm %*% diag(1/col.norm(X_norm)), x1$mat_norm)
    }
})

test_that('matrix_normalize sparse', {
    X_norm <- as(X, "dgCMatrix")
    X_norm_copy = as(X, "dgCMatrix")
    
    expect_equal(X_norm, X_norm_copy)
    
    x1 <- .Call("_L0Learn_R_matrix_normalize_sparse", X_norm)
    
    expect_equal(X_norm, X_norm_copy) # R should not modify X_norm
    
    expect_equal(col.norm(X_sparse), as.vector(x1$ScaleX))
    expect_equal(as.matrix(X_norm %*% diag(1/col.norm(X_sparse))), as.matrix(x1$mat_norm))
})

test_that('matrix_center dense', {
    for (intercept in c(TRUE, FALSE)){
        x_norm <- 0*X
        x1 <- .Call("_L0Learn_R_matrix_center_dense", X, x_norm, intercept)
        
        if (intercept){
            expect_equal(as.vector(x1$MeanX), colMeans(X))
            expect_equal(x1$mat_norm, center_colmeans(X))
        } else {
            expect_equal(as.vector(x1$MeanX), 0*colMeans(X))
            expect_equal(x1$mat_norm, X)
        }
    }
})


test_that('matrix_center sparse', {
    for (intercept in c(TRUE, FALSE)){
        x_norm = 0*X_sparse
        x1 <- .Call("_L0Learn_R_matrix_center_sparse", X_sparse, x_norm, intercept)
            expect_equal(as.vector(x1$MeanX), 0*colMeans(X))
            expect_equal(x1$mat_norm, X_sparse)
    }
})


test_that('vector_subset',{
    v <- 1:n
    indicies <- sample(v, floor(n/3))
    
    x1 <- v[indicies]
    x2 <- .Call("_L0Learn_R_vector_subset", v, indicies-1)
    expect_equal(x1, x2)
    
})

test_that("arg_sort", {
    v <- 0:n
    r <- sample(0:n)
    order1 <- order(r, decreasing=TRUE)
    expect_equal(r[order1], rev(v))
    
    order2 <- .Call("_L0Learn_R_arg_sort", r)
    
    expect_equal(order1, order2+1)
})
