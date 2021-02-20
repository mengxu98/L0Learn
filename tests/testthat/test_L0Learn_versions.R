library("testthat")
library("L0Learn")

L0LEARNVERSIONDATAFOLDER = normalizePath(file.path("~/Documents/GitHub/L0LearnVersionData"))
version_to_load_from="1.2.0"
time_version_to_load_from = paste('no_grid_time', version_to_load_from, sep='_')

current_version = packageVersion("L0Learn")

# Assert data is available to test from
if (!(time_version_to_load_from %in% dir(L0LEARNVERSIONDATAFOLDER))){
    print(L0LEARNVERSIONDATAFOLDER)
    stop("'time_version_to_load_from' must exist in 'L0LEARNVERSIONDATAFOLDER'")
}


test_that("All versions run as expected", {
    # Load data object
    data_large <- readRDS(file.path(L0LEARNVERSIONDATAFOLDER, time_version_to_load_from, "data_large.rData"))
    data_medium <- readRDS(file.path(L0LEARNVERSIONDATAFOLDER, time_version_to_load_from, "data_medium.rData"))
    data_small <- readRDS(file.path(L0LEARNVERSIONDATAFOLDER, time_version_to_load_from, "data_small.rData"))
    
    # L0_grid <- readRDS(file.path(L0LEARNVERSIONDATAFOLDER, time_version_to_load_from, "L0_grid.rData"))
    # L012_grid <- readRDS(file.path(L0LEARNVERSIONDATAFOLDER, time_version_to_load_from, "L012_grid.rData"))
    # 
    # L0_nGamma = 1
    # L012_nGamma <- length(L012_grid)
    
    # Load tests:
    benchmark_tests <- readLines(file.path(L0LEARNVERSIONDATAFOLDER, 
                                           time_version_to_load_from, 'benchmark_tests.txt'))
    
    grid_finding_runs <- readLines(file.path(L0LEARNVERSIONDATAFOLDER, 
                                             time_version_to_load_from, 'grid_finding_runs.txt'))
    
    # Run tests:
    for (i in 1:length(tests)){
        # NOTE: Between v1.2.0 and v1.2.1 a change in standardization results
        # in numerical precision issues on the order of ~1e-5
        
        if ((version_to_load_from == '1.2.0') && (current_version != '1.2.0')){
            tolerance = 1e-5
        } else {
            tolerance = 1e-9
        }
        version_time <- readRDS(file.path(L0LEARNVERSIONDATAFOLDER,
                                          time_version_to_load_from,
                                          paste(i, ".rData", sep='')))
        
        OLD_GRID_FIT <- version_time$fit
        
        GRID_FIT <- eval(parse(text=grid_finding_runs[[i]]))
      
        expect_equal(OLD_GRID_FIT, GRID_FIT$fit, info=i, tolerance=tolerance)
    }
})