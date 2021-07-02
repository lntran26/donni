import dadi

if __name__ == '__main__': 
    func = dadi.Demographics2D.split_mig
    ns = [20,20]
    pts_l = [40, 50, 60]
    func_ex = dadi.Numerics.make_extrap_func(func)
    lower_bound, upper_bound = [1e-2,1e-2,0.1,1], [1e2,1e2,2,10]
    sel_params = [0.8, 1.2, 0.8, 3.5]
    fs = func_ex((1, 1, 0.5, 4), ns, pts_l)
    p0 = dadi.Misc.perturb_params(sel_params, 
                    lower_bound=lower_bound, upper_bound=upper_bound)
    popt, llnlopt = dadi.Inference.opt(p0, fs, func_ex, pts_l, 
                                lower_bound=lower_bound,
                                upper_bound=upper_bound,
                                multinom=True,
                                verbose=5)
    print(popt)
