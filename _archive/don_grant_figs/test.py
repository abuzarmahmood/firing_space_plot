norm = matplotlib.colors.Normalize(0,1)
cmap_object= matplotlib.cm.ScalarMappable(cmap = 'viridis', norm = norm)
alpha = 0.05
min_time = 100
sig_pval_mat = 1*((p_val_mat>(1-(alpha/2))) \
        + (p_val_mat < (alpha/2)))
t_vec = np.arange(p_val_mat.shape[-1])

fig, ax = gen_square_subplots(coherence_boot_array.shape[1])
for ax_num, this_ax in enumerate(ax.flatten()\
        [:coherence_boot_array.shape[1]]):

    diff_vals = np.diff(sig_pval_mat[ax_num])
    change_inds = np.where(diff_vals!=0)[0]
    if diff_vals[change_inds][0] == -1:
        change_inds = np.concatenate([np.array(0)[np.newaxis],change_inds])    
    if diff_vals[change_inds][-1] == 1:
        change_inds = np.concatenate([change_inds, (max(t_vec)-1)[np.newaxis]])    
    change_inds = change_inds.reshape((-1,2))
    fin_change_inds = change_inds[np.diff(change_inds,axis=-1).flatten() > min_time]

    this_coherence = coherence_boot_array[:,ax_num]
    mean_val = np.mean(this_coherence,axis=0)
    std_val = np.std(this_coherence,axis=0)
    t_vec = np.arange(this_coherence.shape[-1])
    this_ax.plot(t_vec,mean_val)
    this_ax.fill_between(\
            x = t_vec,
            y1 = mean_val - 2*std_val,
            y2 = mean_val + 2*std_val, 
            alpha = 0.5)
    this_ax.hlines((lower_bound[ax_num],higher_bound[ax_num]),
            0, coherence_boot_array.shape[-1], color = 'r', alpha = 0.5)
    for interval  in fin_change_inds:
        this_ax.axvspan(interval[0],interval[1],facecolor='y',alpha = 0.5)
    this_ax.set_title(freq_label_list[ax_num])
plt.suptitle('Baseline 95% CI (Bandpass) \n'\
        + "_".join(fin_lfp_node_path_list[this_node_num].split('/')[2:4]) + \
        '\nalpha = {}, minimum significant window  = {}'.format(alpha,min_time))
fig.set_size_inches(16,8)
plt.show()
