for j in range(len(all_pre_stim_window)):
    pre_stim_window = all_pre_stim_window[j]
    all_pre_stim_t = np.arange(2000,200,-pre_stim_window) # Where to start
    for i in range(len(all_pre_stim_t)):
        pre_stim_t = all_pre_stim_t[i]
        corrs = []
        stim_vars = []
        #fig = plt.figure()
        for taste in range(4):
            data = off_firing[taste]
            
            mean_pre_stim = np.mean(data[:,:,int((pre_stim_t - pre_stim_window)/step_size):int(pre_stim_t/step_size)],axis = 2).T #(neurons x trials)
            pre_stim_dist = np.tril(dist_mat(mean_pre_stim,mean_pre_stim)) # Take out upper diagonal to prevent double counting
            
            stim_dat = data[:,:,int(stim_t/step_size):int((stim_t+post_stim_t)/step_size)]
            stim_dists = np.zeros((stim_dat.shape[1],stim_dat.shape[1],stim_dat.shape[2]))
            stim_dist_var = np.zeros(stim_dat.shape[2])
            for time in range(stim_dists.shape[2]):
                stim_dists[:,:,time] = dist_mat(stim_dat[:,:,time].T,stim_dat[:,:,time].T)
                stim_dist_var_temp = np.tril(stim_dists[:,:,time])
                stim_dist_var[time] = np.var(stim_dist_var_temp[stim_dist_var_temp.nonzero()].flatten())
            sum_stim_dists = np.tril(np.sum(stim_dists,axis = 2))
            
            temp_corr = pearsonr(pre_stim_dist[pre_stim_dist.nonzero()].flatten(),sum_stim_dists[sum_stim_dists.nonzero()].flatten())
            temp_corr_dat = pd.DataFrame(dict(file = file, taste = taste, 
                    baseline_end = pre_stim_t, rho = temp_corr[0],p = temp_corr[1],
                    index = [corr_dat.shape[0]], shuffle = False, pre_stim_window_size = pre_stim_window))
            
            for repeats in range(200): # Shuffle trials
                temp_corr_sh = pearsonr(pre_stim_dist[pre_stim_dist.nonzero()].flatten(),
                                                      np.random.permutation(sum_stim_dists[sum_stim_dists.nonzero()].flatten()))
                temp_corr_sh_dat = pd.DataFrame(dict(file = file, taste = taste, 
                        baseline_end = pre_stim_t, rho = temp_corr_sh[0],p = temp_corr_sh[1],
                        index = [corr_dat.shape[0]], shuffle = True, pre_stim_window_size = pre_stim_window))
                corr_dat = pd.concat([corr_dat,temp_corr_sh_dat])
            
            corr_dat = pd.concat([corr_dat,temp_corr_dat])
        print('file %i end_at %i window %i' % (file, pre_stim_t,pre_stim_window))

# Define all relevant windows in the beginning so
# you don't have a million variables floating around

# Also, just convert them to indices
    
baseline_window_sizes = np.arange(100,1000,100)
baseline_window_end = 2000
baseline_window_start = 200
all_baseline_windows = []
for i in range(len(baseline_window_sizes)):
    temp_baseline_windows = np.arange(baseline_window_end,baseline_window_start-baseline_window_sizes[i],-baseline_window_sizes[i])
    temp_baseline_windows = temp_baseline_windows[temp_baseline_windows>0]
    for j in range(0,len(temp_baseline_windows)-1):
        all_baseline_windows.append((temp_baseline_windows[j+1],temp_baseline_windows[j]))

stimulus_time = 2000
stimulus_window_size = 2000
step_size = 25

shuffle_repeats = 1000

for taste in range(4):
    for i in range(len(all_baseline_windows)):
        data = off_firing[taste]
        
        baseline_start = int(all_baseline_windows[i][0]/step_size)
        baseline_end = int(all_baseline_windows[i][1]/step_size)
        stim_start = int(stimulus_time/step_size)
        stim_end = int((stimulus_time + stimulus_window_size)/step_size)
        
        mean_pre_stim = np.mean(data[:,:,baseline_start:baseline_end],axis = 2).T #(neurons x trials)
        pre_stim_dist = np.tril(dist_mat(mean_pre_stim,mean_pre_stim)) # Take out upper diagonal to prevent double counting
        
        stim_dat = data[:,:,stim_start:stim_end]
        stim_dists = np.zeros((stim_dat.shape[1],stim_dat.shape[1],stim_dat.shape[2]))
        for time in range(stim_dists.shape[2]):
            stim_dists[:,:,time] = dist_mat(stim_dat[:,:,time].T,stim_dat[:,:,time].T)
        sum_stim_dists = np.tril(np.sum(stim_dists,axis = 2))
        
        temp_corr = pearsonr(pre_stim_dist[pre_stim_dist.nonzero()].flatten(),sum_stim_dists[sum_stim_dists.nonzero()].flatten())
        temp_corr_dat = pd.DataFrame(dict(file = file, taste = taste, 
                baseline_end = baseline_end*step_size, rho = temp_corr_sh[0],p = temp_corr_sh[1],
                index = [corr_dat.shape[0]], shuffle = False, pre_stim_window_size = (baseline_end - baseline_start)*step_size))
        corr_dat = pd.concat([corr_dat,temp_corr_dat])
        
        pool = mp.Pool(processes = mp.cpu_count())
        results = [pool.apply_async(stim_corr_shuffle, args = (pre_stim_dist, sum_stim_dists)) for repeat in range(shuffle_repeats)]
        output = [p.get() for p in results]
        pool.close()
        pool.join()
        
        for i in range(len(output)):
            corr_dat = pd.concat([corr_dat,output[i]])
        
    def stim_corr_shuffle(pre_stim_dist, sum_stim_dists):
        temp_corr_sh = pearsonr(pre_stim_dist[pre_stim_dist.nonzero()].flatten(),
                                              np.random.permutation(sum_stim_dists[sum_stim_dists.nonzero()].flatten()))
        temp_corr_sh_dat = pd.DataFrame(dict(file = file, taste = taste, 
                baseline_end = baseline_end*step_size, rho = temp_corr_sh[0],p = temp_corr_sh[1],
                index = [corr_dat.shape[0]], shuffle = True, pre_stim_window_size = (baseline_end - baseline_start)*step_size))
        return temp_corr_sh_dat
    