import dadi
import matplotlib.pyplot as pyplot

# specify 1d-2epoch model, sample size and grid size with dadi
func = dadi.Demographics1D.two_epoch
func_ex = dadi.Numerics.make_extrap_func(func)
ns = [20,]
pts_l = [40, 50, 60]

# specify a toy parameter to plot
# this example has T/nu <= 5
nu = 10
T = 0.5
# this example has T/nu > 5
# nu = 0.05
# T = 1.5
# param
param = (nu, T)

# specify theta list
theta_list = [100, 1000, 10000]
fs = func_ex(param, ns, pts_l)
list_fs = [fs] # this should have 4 fs for each theta

for theta in theta_list:
    fs_tostore = (theta*abs(fs)).sample()
    if fs_tostore.sum()!=0:
        fs_norm = fs_tostore/fs_tostore.sum()
    list_fs.append(fs_norm)

print(list_fs)

#plotting
for fs in list_fs:
    dadi.Plotting.plot_1d_fs(fs)
    pyplot.show()