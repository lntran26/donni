import dadi
import pylab

# Add in fixed-theta
def split_mig_fixed_theta(params, ns, pts):
    nu1, nu2, T, m = params
    theta1 = 1000

    xx = dadi.Numerics.default_grid(pts)

    phi = dadi.PhiManip.phi_1D(xx, theta0 = theta1)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.Integration.two_pops(phi, xx, T, nu1, nu2, m12=m, m21=m, theta0 = theta1)

    fs = dadi.Spectrum.from_phi(phi, ns, (xx, xx))
    return fs

# problem params
# p = [10,10,0.5,1] 
# p = [0.24082, 0.14498, 1.35188, 9.29242]
# p = [0.03507, 0.20497, 1.73732, 6.40696]
# can do process_ii% for this as well

# well-behaved params
p = [4.86386, 35.6438, 1.24686, 5.12176]

# designate demographic model, sample size, and extrapolation grid 
# func = dadi.Demographics2D.split_mig
func = split_mig_fixed_theta
ns = [20,20]
pts_l = [40, 50, 60]
func_ex = dadi.Numerics.make_extrap_func(func)

# generate the fs
fs = func_ex(p, ns, pts_l) 

# 400 jobs --> rocess_ii%4 command run 100 times for each of the 4 sel_params
# sel_params = [[1,2,3,4],[],[],[]][process_ii%4]

# sel_params = [0.1, 1, 1.6, 8]
# sel_params = [9, 11, 1.6, 1]
# sel_params = [0.1, 1, 1.6, 1]
# sel_params = [50, 2, 1.6, 2]
# sel_params = [1, 1, 0.95, 4.5]
# sel_params = [0.05, 0.5, 1, 8]
sel_params = [0.05, 0.5, 1, 5]

lower_bound, upper_bound = [1e-2,1e-2,0.1,1], [1e2,1e2,2,10]

# perturb param
p0 = dadi.Misc.perturb_params(sel_params, lower_bound=lower_bound,
                              upper_bound=upper_bound)

# select optimizer to run optimization with
# popt = dadi.Inference.optimize_log(p0, fs, func_ex, pts_l,      lower_bound=lower_bound, upper_bound=upper_bound, multinom=True, 
# verbose=5) #fixed_params=[None,None,0.5,None]

popt, llnlopt = dadi.Inference.opt(p0, fs, func_ex, pts_l, 
                                    lower_bound=lower_bound,
                                    upper_bound=upper_bound,
                                    multinom=True,
                                    verbose=5)

# make inferred fs from popt
fs_inferred = func_ex(popt, ns, pts_l) 
# # likelihood calculation when using old optimizer
# ll = dadi.Inference.ll_multinom(fs_inferred fs) 
# # multinom rescale things but here it doesn't seem to affect anything

# this calculate the "true" likelihood value
ll_true = round(dadi.Inference.ll(fs, fs),5)
    
print("Problem params:", p)
print("Selected params:", sel_params) # more standard to use this than using perturbed params
# print("Perturbed params:", p0)
print("Optimized params:", end = " ")
popt = [round(p,5) for p in popt]
print(*popt, sep=', ')

# print("Perturbed params:", p0)
# print("Max Log Likelihood:", ll)
print("Max Log Likelihood:", round(llnlopt,5))
print("True Log Likelihood:", ll_true)
    
dadi.Plotting.plot_2d_comp_multinom(fs, fs_inferred)
pylab.show()

# # Note: Had data and model swapped in these plots
# # dadi.Plotting.plot_2d_comp_multinom(fs, fs_inferred)
# snm_func_ex = dadi.Numerics.make_extrap_func(dadi.Demographics2D.snm)
# fs_snm = snm_func_ex([], ns, pts_l)

# print('LL of data with inferred: {0}'.format(dadi.Inference.ll_multinom(fs_inferred, fs)))
# print('LL of data with snm: {0}'.format(dadi.Inference.ll_multinom(fs_snm, fs)))

# dadi.Plotting.plot_2d_comp_multinom(fs_snm, fs, resid_range = 10,fig_num=2)
# pylab.show()