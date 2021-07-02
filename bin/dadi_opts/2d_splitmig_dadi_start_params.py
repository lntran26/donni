import pickle

test_set = pickle.load(open('test-data-corrected-2','rb'))[2]
p_true = list(test_set.keys())[:50] # log scale
fs_data = [1000*test_set[p] for p in p_true] # matching fs


p1 = [10**-1.5, 10**1, 0.5, 2]
p2 = [10**1, 10**-1.5, 1.5, 6]
p3 = [10**1.2, 10**1.2, 1, 8]
p4 = [10**-0.5, 10**0.5, 1.8, 3]
p5 = [10**0.9, 10**0.9, 1, 4]

p_list = [p1, p2, p4, p4, p5]

# list of 50 lists, length 5 for each starting p
popt_list = pickle.load(open('dadi_opt_results_corrected','rb'))

start = []
# get the max likelihood
for l in popt_list:
    max_i = 0
    max_ll = l[0][1]
    for i in range(1, len(l)):
        if l[i][1] > max_ll:
            max_ll = l[i][1]
            max_i = i
    start.append(p_list[max_i])

pickle.dump(start, open('dadi_start', 'wb'), 2)
pickle.dump(fs_data, open('dadi_fs_data', 'wb'), 2)
pickle.dump(p_true, open('dadi_p_true', 'wb'), 2)

