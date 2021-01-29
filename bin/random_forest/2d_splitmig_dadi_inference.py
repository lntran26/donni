import dadi
import pylab

p = [10,10,0.5,1] # problem param
# can do process_ii% for this as well

# designate demographic model, sample size, and extrapolation grid 
func = dadi.Demographics2D.split_mig
ns = [5,5]
pts_l = [40, 50, 60]
func_ex = dadi.Numerics.make_extrap_func(func)

# generate the fs
fs = func_ex(p, ns, pts_l) 

# fs = fs/fs.sum() # norm

# 400 jobs --> run 100 for each sel_params
sel_params = [[1,2,3,4],[],[],[]][process_ii%4]

# sel_params = [0.1, 1, 1.6, 8]
# sel_params = [9, 11, 1.6, 1]
# sel_params = [0.1, 1, 1.6, 1]
sel_params = [50, 2, 1.6, 2]
lower_bound, upper_bound = [1e-2,1e-2,0.1,1], [1e2,1e2,2,10]

p0 = dadi.Misc.perturb_params(sel_params, lower_bound=lower_bound,
                              upper_bound=upper_bound)
popt = dadi.Inference.optimize_log(p0, fs, func_ex, pts_l,      lower_bound=lower_bound, upper_bound=upper_bound, multinom=True, 
verbose=True, fixed_params=[None,None,0.5,None])

# popt, llnlopt = dadi.Inference.opt(p0, fs, func_ex, pts_l, 
#                                     lower_bound=lower_bound,
#                                     upper_bound=upper_bound,
#                                     verbose=0)

# make model from popt
model = func_ex(popt, ns, pts_l) 
# likelihood
ll = dadi.Inference.ll_multinom(model, fs)
    
print("Original params:", p)
print("Optimized params:", popt)
print("Selected params:", sel_params) # more standard to use this than using perturbed params
# print("Perturbed params:", p0)
print("Max Likelihood:", ll)
    
dadi.Plotting.plot_2d_comp_multinom(fs, model)
pylab.show()