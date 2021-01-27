import dadi
import pylab

p = [10,10,0.5,1] # problem param

# designate demographic model, sample size, and extrapolation grid 
func = dadi.Demographics2D.split_mig
ns = [20,20]
pts_l = [40, 50, 60]
func_ex = dadi.Numerics.make_extrap_func(func)

# generate the fs
fs = func_ex(p, ns, pts_l) 
fs = fs/fs.sum() # norm
  
# sel_params = [0.1, 1, 1.6, 8]
# sel_params = [9, 11, 1.6, 1]
# sel_params = [0.1, 1, 1.6, 1]
sel_params = [50, 2, 1.6, 2]
lower_bound, upper_bound = [1e-2,1e-2,0.1,1], [1e2,1e2,2,10]

p0 = dadi.Misc.perturb_params(sel_params, lower_bound=lower_bound,
                              upper_bound=upper_bound)
popt = dadi.Inference.optimize_log(p0, fs, func_ex, pts_l, lower_bound=lower_bound,
                                   upper_bound=upper_bound, multinom=True)
    
print("Original params: ", p)
print("Optimized params:", popt)

print("Perturbed params:", p0)

fs_inferred = func_ex(popt, ns, pts_l)
    
dadi.Plotting.plot_2d_comp_multinom(fs, fs_inferred)
pylab.show()