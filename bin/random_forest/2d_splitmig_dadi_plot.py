import dadi
import pylab

# True param
## CASE OF TRAIN ON THETA=1 TEST ON THETA=1000
# p_true = [10**-0.4886980160602463, 10**-0.5986940353641503, 1.757468910280852, 0.19453148795341718] 
# p_true = [10**-0.3062429578920649, 10**-0.5490926066567567, 0.1935590316509991, 0.8709109622898537]
# p_true = [10**-1.2097656295583312, 10**-1.9351137057181433, 0.5543724416104007, 8.31264949890994]
# p_true = [10**-1.5833082424113987, 10**-1.7898266353466883, 1.639787520625167, 6.605239254959342]

## CASE OF TRAIN ON THETA=1 TEST ON THETA=1
# p_true = [10**-0.4515534920900932, 10**-0.6408586386095445, 1.9224734379044333, 0.40008275245460667]
# p_true = [10**-1.8126667771094698, 10**1.1089117388892862, 1.7511022025815246, 0.1532328791271397]
# p_true = [10**1.9748695063515624, 10**-1.5188754701940304, 0.24183445250383992, 3.1029724104877596]
p_true = [10**0.02723128080176984, 10**-1.9516108419358908, 1.9100103765652285, 1.136902024957365]

# Predict param
## CASE OF TRAIN ON THETA=1 TEST ON THETA=1000
# p_pred = [10**-1.64444444, 10**-1.63555556, 1.413375, 3.3175]
# p_pred = [10**-0.66222222, 10**-0.93333333, 1.0975, 4.8475]
# p_pred = [10**-1.00888889, 10**-1.54666667, 1.29383333, 2.2375]
# p_pred = [10**-1.33777778, 10**-1.40444444, 1.1355, 2.1475]

## CASE OF TRAIN ON THETA=1 TEST ON THETA=1
# p_pred = [10**-1.72888889, 10**-1.85777778, 1.17636131, 7.84]
# p_pred = [10**-1.80888889, 10**0.84, 0.84575, 1.0225]
# p_pred = [10**1.09333333, 10**-1.72888889, 0.93125, 6.2425]
p_pred = [10**-0.07555556, 10**-1.94222222, 1.01675, 1.5175]

# designate demographic model, sample size, and extrapolation grid 
func = dadi.Demographics2D.split_mig
ns = [20,20]
pts_l = [40, 50, 60]
func_ex = dadi.Numerics.make_extrap_func(func)

# generate the fs
fs_true = func_ex(p_true, ns, pts_l) 
fs_pred = func_ex(p_pred, ns, pts_l)
    
dadi.Plotting.plot_2d_comp_multinom(fs_true, fs_pred)

pylab.show()