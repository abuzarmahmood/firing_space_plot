import numpy as np
from scipy import stats
from scipy.stats import percentileofscore as p_of_s
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm, trange 
import itertools as it

def parallelize(func, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

def corr_percentile_single(a,b, shuffles = 1000):
    """
    Calculates correlation, shuffled correlations, and percentile
    of correlation relative to shuffled distribution

    Input:
        a,b : Vectors with length = trials
        shuffles : Number of shuffles to perform

    Output:
        percentile_val : Percentile of actual corr relative to shuffle dist
        corr_val : Rho value of the actual correlation
        shuffle_vals : List of rho values of shuffled correlations
    """
    corr_val = stats.spearmanr(a,b)[0]
    shuffle_vals = [stats.spearmanr(a, 
                    np.random.permutation(b))[0] \
            for i in range(shuffles)]
    percentile_val = p_of_s(shuffle_vals, corr_val)
    return percentile_val, corr_val, shuffle_vals

def return_corr_percentile(tau_array, shuffles = 5000):
    """
    Calcualtes outputs of "corr_percentile_single" for all
    transitions present in a dataset

    Input:
        tau_array : Shape :: regions x transition x trials
        shuffles : Number of shuffles to perform

    Output:
        ** All outputs will be for all-to-all transition comparisons
        ** User may choose to keep only the matched transitions if needed
        percentile_array : Shape :: transitions x transitions
        corr_array : Shape :: transitions x transitions
        shuffle_array : Shape :: transitions x transitions x shuffles
    """
    trans_list = np.arange(tau_array.shape[1])
    # **Note: The transitions in BLA and GC are not the same,
    #           therefore we must look at all permutations, not simply
    #           all combinations.
    comparison_list = list(it.product(trans_list, trans_list))
    percentile_array = np.zeros((tau_array.shape[1], tau_array.shape[1]))
    corr_array = np.zeros((tau_array.shape[1], tau_array.shape[1]))
    shuffle_array = np.zeros((tau_array.shape[1], tau_array.shape[1], shuffles))
    for this_comp in tqdm(comparison_list):
        percentile_val, corr_val, shuffle_vals = \
                corr_percentile_single(tau_array[0, this_comp[0]],
                                        tau_array[1, this_comp[1]],
                                        shuffles = shuffles)
        percentile_array[this_comp] = percentile_val
        corr_array[this_comp] = corr_val
        shuffle_array[this_comp] = shuffle_vals
    return percentile_array, corr_array, shuffle_array

def parallel_return_corr_percentile(tau_list):
    """
    Parallizes calculation of "return_corr_percentile" over
    multiple datasets

    Inputs:
        tau_list : List containing multiple tau_array's with shape
                    as needed by return_corr_percentile

    Outputs:
        List of outputs of return_corr_percentile
        i.e. each element of the list will contain 3 arrays
        
    """
    return parallelize(return_corr_percentile, tau_list)


########################################
## Run test example
########################################
def gen_tau_array(trials = 30, transitions = 3):
    x = np.random.random((trials, transitions, 1))
    tau_array = np.concatenate([x,x],axis=-1).T
    return tau_array

# Make sure corr_percentile_single works
perc, corr, shuff = corr_percentile_single(*tau_array[:,0])

perc, corr, shuff = return_corr_percentile(tau_array)
# perc should show high percentiles for only matched transitions
print(perc)

tau_list = [gen_tau_array() for i in range(10)]
outs = parallel_return_corr_percentile(tau_list) 
