####################################
## Lining up by state transitions ##
####################################

# Import libraries, change folder, read file, make plot directory
import os
import tables
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt
from scipy.stats import pearsonr
import re
import easygui
import shutil
import matplotlib.image as mpimg
from scipy.stats import rankdata

if easygui.ynbox('Load directory list from file?', 'Title', ('Yes', 'No')):
    dir_list_folder = easygui.diropenbox('Select folder containing directory list')
    f = open(dir_list_folder + '/' + 'all_dirs', 'r')
    dir_list = []
    for line in f.readlines():
        dir_list.append(line.strip())
    f.close()
    
        #Specify params or load from file
    if easygui.ynbox('Load params from file?', 'Title', ('Yes', 'No')):
        params_folder = easygui.diropenbox('Select parameters folder')
        params_file = 'correlation_lining_params'
        f = open(params_folder+'/'+params_file, 'r')
        params = []
        for line in f.readlines():
            params.append(line)
        f.close()
        min_t = int(params[0])
        max_t = int(params[1])
        state_sig = float(params[2])
        palatability_order = [int(i) for i in params[3:7]]
        pre_stim_t = int(params[7]) #2000   #time before stimulus
        data_window_pre = int(params[8]) #500   #window before transition time
        data_window_post = int(params[9]) #750  #window after transition time
        bin_window_size = int(params[10]) #250
        step_size = int(params[11]) #25
        min_transition = int(params[12]) #250
        max_transition = int(params[13]) #1750
        hmm_bin_size = int(params[14]) #10

    else:   #Request for parameters and save to file in correlation plot folder
        lining_params = easygui.multenterbox('Parameters for correlation',fields = ['Start time for palatability', 'End time for palatability', 'Probability to treat state as significant'])
        min_t = int(lining_params[0])
        max_t = int(lining_params[1])
        state_sig = float(lining_params[2])

        #palatability_order = [3,4,1,2] #Manually entered, corresponds to dig_in [0, 1, 2, 3]
        palatability_order = easygui.multenterbox('Palatability rank for all dig_ins',fields = ['dig_in_0','dig_in_1','dig_in_2','dig_in_3'])
        palatability_order = [int(i) for i in palatability_order]

        #Params for making PSTH's
        psth_params = easygui.multenterbox('Parameters for making PSTHs',fields = ['Pre-stimulus time (2000)','Time wanted before transition (500)', \
            'Time watned after transition (750)', 'Binning window size (250)', 'Window step size (25)', 'First Transition Cutoff Time (250)', \
            'Final Transition Cutoff time (1750)', 'HMM Bin Size (10)'])
        pre_stim_t = int(psth_params[0]) #2000   #time before stimulus
        data_window_pre = int(psth_params[1]) #500   #window before transition time
        data_window_post = int(psth_params[2]) #750  #window after transition time
        bin_window_size = int(psth_params[3]) #250
        step_size = int(psth_params[4]) #25

        min_transition = int(psth_params[5]) #250
        max_transition = int(psth_params[6]) #1750
        hmm_bin_size = int(psth_params[7]) #10

        if easygui.ynbox('Write parameters to file?', 'Title', ('Yes', 'No')):
            if os.path.isfile('correlation_lining_params'):
                if easygui.ynbox('Overwrite old parameters file?', 'Title', ('Yes', 'No')):
                    f = open('correlation_lining_params', 'w')
                    for params in lining_params:
                        	print(params, file=f)
                    for params in palatability_order:
                        	print(params, file=f)
                    for params in psth_params:
                        	print(params, file=f)
                    f.close()

    for data_direc in dir_list:
        #data_direc = easygui.diropenbox('Choose directory to pick hdf5 file from')
        os.chdir(data_direc)

        # Look for the hdf5 file in the directory
        file_list = os.listdir('./')
        hdf5_name = ''
        for files in file_list:
        	if files[-2:] == 'h5':
        		hdf5_name = files

        # Open the hdf5 file
        hf5 = tables.open_file(hdf5_name, 'r+')

        plot_dir = './correlation_analysis_plots'
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
            
        transition_check_dir = './transition_check_dir'
        if not os.path.isdir(transition_check_dir):
            os.mkdir(transition_check_dir)

        # For extracting spike trains
        dig_in = hf5.list_nodes('/spike_trains')
        dig_in = [dig_in[i] if dig_in[i].__str__().find('dig_in')!=-1 else None for i in range(len(dig_in))]
        dig_in = list(filter(None, dig_in))

        ## Number of trials per taste for indexing when concatenating spike trains
        taste_n = [dig_in[i].spike_array[:].shape[0] for i in range(len(dig_in))]
        if np.std(taste_n) == 0:
            taste_n = taste_n[0]
        else:
            taste_n = int(easygui.multenterbox('How many trails per taste??',fields = ['# of trials'])[0])
        
        # Delete and remake these dirs to make sure they are empty for new analysis                
        try: 
            shutil.rmtree(plot_dir)
            os.mkdir(plot_dir)
        except:
            print('Could not delete correlation plots folder!')  
            
        try: 
            shutil.rmtree(transition_check_dir)
            os.mkdir(transition_check_dir)
        except:
            print('Could not delete transition check plot folder!')  

        ################
        ## OFF TRIALS ##
        ################
        # Find number of states to use in extracting from hdf5 file
        all_states = hf5.list_nodes('/spike_trains/multinomial_hmm_results/laser_off')

        all_states = re.findall('\d',all_states.__str__())
        for i in range(len(all_states)):
            all_states[i] = int(all_states[i])


        for state in all_states: ## Loop through all models in HMM
            exec('post_prob = hf5.root.spike_trains.multinomial_hmm_results.laser_off.states_%i.posterior_proba[:]' % state)
            #post_prob shape = [trials,time,states]
            exec('t = np.array(hf5.root.spike_trains.multinomial_hmm_results.laser_off.states_%i.time[:])' % state)
            indices = np.where((t >= min_t)*(t <= max_t))[0]

            # Delete the r_pearson and p_pearson nodes if they already exist (this ensures there is not
            # output if state transitions do not meet cutoff criteria)
            try:
                hf5.remove_node('/spike_trains/multinomial_hmm_results/laser_off/states_%i' % state, 'r_pearson')
                hf5.remove_node('/spike_trains/multinomial_hmm_results/laser_off/states_%i' % state, 'p_pearson')
                hf5.remove_node('/spike_trains/multinomial_hmm_results/laser_off/states_%i' % state, 'palatability_state_transition_t')
                hf5.flush()
            except:
                pass

            ## Note : HMM model is fit to particular trials but predicts posterior
            ## probabilities for transitions in all trials.
            ## Only using trials for which the model was fit i.e off_trials for off model and on_trials for on model.

            # Which trials did NOT have laser
            off_trials = np.concatenate(tuple(np.where(dig_in[i].laser_durations[:] == 0)[0] + taste_n*i for i in range(len(dig_in))))

            post_prob = post_prob[off_trials,:,:]
            ## Since we're indexing post_prob by off_trials, EVERYTHING has to be indexed by off_trials

            sum_state_mass = np.sum(post_prob[:, indices, :], axis = (0,1)) #to find index of state with highest occurence during set time
            #transition = np.where(post_prob[:,:,np.argmax(sum_state_mass)] >= 0.8) #where[0] = x = trial, where[1] = y = time
            max_state = np.argmax(sum_state_mass)

            # Find which trials have transitions and first point of transition
            # Find trials where transition happened after the minimum cutoff
            # For these trials, find when the previous transition happened; if none, set it equal to the transition time.
            # WELCOME TO 'IF' STATEMENT HELL!!
            transition = []
            prev_transition = []
            for i in range(post_prob.shape[0]):
                this_transition = np.where(post_prob[i, :, max_state] >= state_sig)[0]
                all_prev_transition = np.where(post_prob[i, :, np.arange(state)!=max_state] >= state_sig)[1] # Transition of all other states
                if len(this_transition) > 0:
                    if (this_transition[0]>=(min_transition/hmm_bin_size))*(this_transition[0]<=(max_transition/hmm_bin_size)): #Transition time agrees with restrictions
                        transition.append([i, this_transition[0]])
                        if len(all_prev_transition[all_prev_transition < this_transition[0]])>0:
                            prev_transition.append(max(all_prev_transition[all_prev_transition < this_transition[0]]))
                        else:
                            prev_transition.append(this_transition[0])
                            # Give them the same value so that the array has the same size
                            # for subtraction for del_transition
                            

            if len(transition)>0:
                
            # If there are transitions, do further processing, otherwise just create empty arrays
            # and write those to files
                del_transition = np.multiply([transition[i][1]-prev_transition[i] for i in range(len(transition))],hmm_bin_size)
                
                # Annotate HMM plots with transition times for checking
                orig_trial_index = [off_trials[i[0]] for i in transition]
                
# =============================================================================
#                 for i in range(len(transition)):
#                     fig = plt.figure(figsize=(16,6))
#                     ax0 = fig.add_subplot(1,2,1)
#                     ax0.plot(np.multiply(range(post_prob.shape[1]),10),post_prob[transition[i][0]]) #Pick out trials from post_prob with transitions
#                     ax0.scatter(transition[i][1]*10,state_sig)
#                     ax0.scatter(prev_transition[i]*10,state_sig)
#                     plt.title('Transition time = %i' % del_transition[i])
#                     hmm_plot_dir = './HMM_plots/Multinomial/laser_off/states_%i' % state
#                     files_list = os.listdir(hmm_plot_dir)
#                     # Pick out HMM plots corresponding to trials with transitions
#                     file_name = files_list[np.where(np.array([file_name.find('Trial_%i.' % (orig_trial_index[i]+1)) for file_name in files_list])>0)[0][0]]
#                     ax1 = fig.add_subplot(1,2,2)
#                     img = mpimg.imread(hmm_plot_dir + '/' + file_name)
#                     ax1.imshow(img)
#                     plt.tight_layout()
#                     fig.savefig(transition_check_dir + '/' + 'off_state%i_trial%i' % (state,orig_trial_index[i]+1))
#                     plt.close(fig)
# =============================================================================
                    
                del_transition = del_transition[del_transition!=0] # Remove all zeros so they don't get averaged
                transition_trials = np.array([transition[i][0] for i in range(len(transition))]) # For indexing
                fin_off_trials = off_trials[transition_trials] # Since we're subsetting off trials and then only trials with transitions
    
    
                #Make palatability vector for ALL trials
                trial_palatability = np.zeros(len(palatability_order)*taste_n)
                for i in range(0,len(palatability_order)):
                    trial_palatability[i*taste_n:(i+1)*taste_n] = np.repeat(palatability_order[i],taste_n)[:]
    
                # Get the spike array for each individual taste and merge all:
                spikes = np.concatenate(tuple(dig_in[i].spike_array[:] for i in range(len(dig_in))), axis = 0)
                off_spikes = spikes[off_trials,:,:] #Index trials with no laser
    
                # Take spikes from trials with transitions and align them according to state transition per trial
                lined_spikes = []
                for trial, time in transition: # Here off_spikes get indexed by transition so it comes to the same index as fin_off_trials
                    lined_spikes.append(off_spikes[trial, :, pre_stim_t + t[time] - data_window_pre :pre_stim_t + t[time] + data_window_post])
                # lined_spikes shape = list of trials, every index = array[neurons , time]
                lined_spikes = np.array(lined_spikes) # [trials, neurons, time]
    
    
                ## Making PSTHs for the spikes
                lined_firing = np.zeros((lined_spikes.shape[0], lined_spikes.shape[1], int((data_window_pre + data_window_post - bin_window_size)/step_size) + 1))
    
                for i in range(lined_firing.shape[0]):
                    for j in range(lined_firing.shape[1]):
                        for k in range(lined_firing.shape[2]):
                            lined_firing[i, j, k] = np.mean(lined_spikes[i, j, step_size*k:step_size*k + bin_window_size])

###########################################                
                ## Sanity check: at transition, there should be a signficant change in firing pattern
                ## compared to unaligned trials
                
                ## for making raster plots if ever needed 
                #lined_spikes_re = lined_spikes.reshape(lined_spikes.shape[0]*lined_spikes.shape[1],lined_spikes.shape[2])
                #for nrn in range(lined_spikes_re.shape[0]):
                #    x = np.where(lined_spikes_re[nrn,:]==1)
                #    for spike in range(len(x)):
                #        plt.vlines(x,nrn,nrn+1)
                nrn_dec = []
                nrn_inc = []
                #nrn_rank_vec = np.zeros(lined_firing_re.shape[0])
                lined_firing_re = lined_firing.reshape(lined_firing.shape[0]*lined_firing.shape[1],lined_firing.shape[2])
                
                for nrn in range(lined_firing_re.shape[0]):
                    if np.sum(lined_firing_re[nrn,:int(np.floor(lined_firing_re.shape[1]/2))]) > np.sum(lined_firing_re[nrn,int(np.floor(lined_firing_re.shape[1]/2)):]):
                        nrn_dec.append((lined_firing_re[nrn,:]-np.mean(lined_firing_re[nrn,:])))#/np.std(lined_firing_re[nrn,:]))
                    else:
                        nrn_inc.append((lined_firing_re[nrn,:]-np.mean(lined_firing_re[nrn,:])))#/np.std(lined_firing_re[nrn,:]))
                    #first_half = np.sum(lined_firing_re[nrn,:int(np.floor(lined_firing_re.shape[1]/2))])
                    #second_half = np.sum(lined_firing_re[nrn,int(np.floor(lined_firing_re.shape[1]/2)):])
                    #nrn_rank_vec[nrn] = first_half/second_half
                    
                #nrn_rank_vec[np.where(np.isnan(nrn_rank_vec)+np.isinf(nrn_rank_vec))] = 0
                #nrn_ranks = rankdata(nrn_rank_vec,method='ordinal')
                #lined_firing_fin = np.array([lined_firing_re[i,:] for i in nrn_ranks-1])
                #plt.imshow(lined_firing_re)
                
                nrn_dec = np.array(nrn_dec)
                nrn_inc = np.array(nrn_inc)
                all_nrn = np.concatenate((nrn_dec,nrn_inc),axis=0)
                
                fir_plots = plt.figure()
                plt.subplot(121)
                plt.imshow(all_nrn/np.nanmax(all_nrn))
                plt.title('Aligned')
                
                # Unaligned spikes
                mid_point = 900
                off_trans = off_spikes[transition_trials,:]
                off_scram = off_trans.reshape(off_trans.shape[0]*off_trans.shape[1],off_trans.shape[2])
                off_scram_fin = off_scram[:,(pre_stim_t + mid_point - data_window_pre):(pre_stim_t + mid_point + data_window_post)]
                
                off_firing = np.zeros((off_scram_fin.shape[0], int((data_window_pre + data_window_post - bin_window_size)/step_size) + 1))
                

                for i in range(off_firing.shape[0]):
                    for j in range(off_firing.shape[1]):
                            off_firing[i, j] = np.mean(off_scram_fin[i, step_size*j:step_size*j + bin_window_size])
                            
                scram_dec = []
                scram_inc = []
                all_scram_std = np.std(lined_firing_re,axis=1)
                all_scram_std[all_scram_std == 0] = np.mean(all_scram_std)
                for nrn in range(off_firing.shape[0]):
                    if np.sum(off_firing[nrn,:int(np.floor(off_firing.shape[1]/2))]) > np.sum(off_firing[nrn,int(np.floor(off_firing.shape[1]/2)):]):
                        scram_dec.append((off_firing[nrn,:]-np.mean(off_firing[nrn,:])))#/np.std(lined_firing_re[nrn,:]))
                    else:
                        scram_inc.append((off_firing[nrn,:]-np.mean(off_firing[nrn,:])))#/np.std(lined_firing_re[nrn,:]))
                        
                scram_dec = np.array(scram_dec)
                scram_inc = np.array(scram_inc)
                all_scram_nrn = np.concatenate((scram_dec,scram_inc),axis=0)
                
                #plt.figure(2)
                plt.subplot(122)
                plt.imshow(all_scram_nrn/np.nanmax(all_scram_nrn))
                plt.title('Unligned')
                plt.tight_layout()
                
                plt.suptitle('Off-firing state %i' % state)
                
                fir_plots.savefig(plot_dir + '/off_firing_rate%i.png' % state)
                plt.close(fir_plots)
                
                ##Cumulative firing plot to go along with the heatmap
                sum_plots = plt.figure()
                plt.subplot(121)
                sum_firing_aligned = np.sum(all_nrn,axis=0)
                plt.plot(sum_firing_aligned)
                plt.title('Aligned')
                plt.subplot(122)
                sum_firing_unaligned = np.sum(all_scram_nrn,axis=0)
                plt.plot(sum_firing_unaligned)
                plt.title('Unaligned')
                plt.suptitle('off-firing cumulative state %i' % state)
                sum_plots.savefig(plot_dir + '/off_firing_rate_sum%i.png' % state)
                plt.close(sum_plots)

###########################################  
                        
                # Plot normalized PSTH of aligned spikes (just for visual inspection)
                test = lined_firing
                test = np.sum(test,0)
                for i in range(test.shape[0]):
                    test[i,:] = np.log2(test[i,:]/test[i,0])
    
                fig = plt.figure()
                plt.title('Aligned firing rates, Off trials , Model states =' + str(state))
                for i in range(test.shape[0]):
                    plt.plot(np.arange(-data_window_pre,data_window_post-bin_window_size+step_size,step_size),test[i,:])
                plt.xlabel('State transition time')
                plt.ylabel('Normalized Firing Rate')
                fig.savefig(plot_dir + '/off_firing_states%i.png' % state)
                plt.close(fig)
    
                # Palatability coefficient used for analysis
                palatability = trial_palatability[fin_off_trials]
    
                # Calculate Pearson parameters and make figures
                r_pearson = np.zeros((lined_firing.shape[1], lined_firing.shape[2])) #shape = [neurons, time]
                p_pearson = np.ones((lined_firing.shape[1], lined_firing.shape[2]))
    
                for i in range(r_pearson.shape[0]):
                    for j in range(r_pearson.shape[1]):
                        r_pearson[i, j], p_pearson[i, j] = pearsonr(lined_firing[:, i, j], palatability)
                        if np.isnan(r_pearson[i, j]):
                            r_pearson[i, j] = 0.0
                            p_pearson[i, j] = 1.0
                            
                pearson_max = np.max(np.mean(r_pearson[:, :]**2, axis = 0))
                pearson_min = np.min(np.mean(r_pearson[:, :]**2, axis = 0))
                pearson_min_max = pearson_max - pearson_min
                fig = plt.figure()
                plt.title('Correlation, Off trials , Model states =' + str(state) + ', Trials = ' + str(len(transition)))
                plt.plot(np.arange(-data_window_pre,data_window_post-bin_window_size+step_size,step_size),np.mean(r_pearson[:, :]**2, axis = 0))
                plt.xlabel('State transition time')
                plt.ylabel('Pearson Correlation Coefficient')
                #ax = fig.add_subplot(111)
                #ax.text(-400,pearson_min+(pearson_min_max*0.9),'Mean state transition \n %i +\- %i msec' % (np.mean(del_transition),np.std(del_transition)))
                fig.savefig(plot_dir + '/off_correlation_states%i.png' % state)
                plt.close(fig)

            else:
                r_pearson = []
                p_pearson = []
                del_transition = []
                fig = plt.figure()
                plt.title('Correlation, Off trials , Model states =' + str(state) + ', Trials = 0')
                plt.plot(0)
                plt.xlabel('State transition time')
                plt.ylabel('Pearson Correlation Coefficient')
                #ax = fig.add_subplot(111)
                #ax.text(-400,pearson_min+(pearson_min_max*0.9),'Mean state transition \n %i +\- %i msec' % (np.mean(del_transition),np.std(del_transition)))
                fig.savefig(plot_dir + '/off_correlation_states%i.png' % state)
                plt.close(fig)
                
            hf5.create_array('/spike_trains/multinomial_hmm_results/laser_off/states_%i' % state, 'r_pearson', r_pearson)
            hf5.create_array('/spike_trains/multinomial_hmm_results/laser_off/states_%i' % state, 'p_pearson', p_pearson)
            hf5.create_array('/spike_trains/multinomial_hmm_results/laser_off/states_%i' % state, 'palatability_state_transition_t', del_transition)
            hf5.flush()


#######################
####### ON TRIALS #####
#######################
            # Code here is same as that for off trials - please refer to comments there for any questions - Thx! (Abu)
        for state in all_states:
            ## Loop through states
            exec('post_prob = hf5.root.spike_trains.multinomial_hmm_results.laser_on.states_%i.posterior_proba[:]' % state) #shape = [trials,time,states]
            exec('t = np.array(hf5.root.spike_trains.multinomial_hmm_results.laser_on.states_%i.time[:])' % state)
            indices = np.where((t >= min_t)*(t <= max_t))[0]

            # Delete the r_pearson and p_pearson nodes if they already exist (this ensures there is not
            # output if state transitions do not meet cutoff criteria)
            try:
                hf5.remove_node('/spike_trains/multinomial_hmm_results/laser_on/states_%i' % state, 'r_pearson')
                hf5.remove_node('/spike_trains/multinomial_hmm_results/laser_on/states_%i' % state, 'p_pearson')
                hf5.remove_node('/spike_trains/multinomial_hmm_results/laser_on/states_%i' % state, 'palatability_state_transition_t')
                hf5.flush()
            except:
                pass

            ## Note : HMM model is fit to particular trials but predicts posterior
            ## probabilities for transitions in all trials.

            on_trials = np.concatenate(tuple(np.where(dig_in[i].laser_durations[:] > 0)[0] + taste_n*i for i in range(len(dig_in))))

            post_prob = post_prob[on_trials,:,:]
            ## Since we're indexing post_prob by off_trials, EVERYTHING has to be indexed by off_trials

            sum_state_mass = np.sum(post_prob[:, indices, :], axis = (0,1)) #to find index of state with highest occurence during set time
            #transition = np.where(post_prob[:,:,np.argmax(sum_state_mass)] >= 0.8) #where[0] = x = trial, where[1] = y = time
            max_state = np.argmax(sum_state_mass)

            # Find which trials have transitions and first point of transition
            transition = []
            prev_transition = []
            for i in range(post_prob.shape[0]):
                this_transition = np.where(post_prob[i, :, max_state] >= state_sig)[0]
                all_prev_transition = np.where(post_prob[i, :, np.arange(state)!=max_state] >= state_sig)[1]
                if len(this_transition) > 0:
                    if (this_transition[0]>=(min_transition/hmm_bin_size))*(this_transition[0]<=(max_transition/hmm_bin_size)):
                        transition.append([i, this_transition[0]])
                        if len(all_prev_transition[all_prev_transition < this_transition[0]])>0:
                            prev_transition.append(max(all_prev_transition[all_prev_transition < this_transition[0]]))
                        else:
                            prev_transition.append(this_transition[0])

            if len(transition)>0:
                del_transition = np.multiply([transition[i][1]-prev_transition[i] for i in range(len(transition))],hmm_bin_size)
                
                orig_trial_index = [on_trials[i[0]] for i in transition]
                
# =============================================================================
#                 for i in range(len(transition)):
#                     fig = plt.figure(figsize=(16,6))
#                     ax0 = fig.add_subplot(1,2,1)
#                     ax0.plot(np.multiply(range(post_prob.shape[1]),10),post_prob[transition[i][0]]) #Pick out trials from post_prob with transitions
#                     ax0.scatter(transition[i][1]*10,state_sig)
#                     ax0.scatter(prev_transition[i]*10,state_sig)
#                     plt.title('Transition time = %i' % del_transition[i])
#                     hmm_plot_dir = './HMM_plots/Multinomial/laser_on/states_%i' % state
#                     files_list = os.listdir(hmm_plot_dir)
#                     # Pick out HMM plots corresponding to trials with transitions
#                     file_name = files_list[np.where(np.array([file_name.find('Trial_%i.' % (orig_trial_index[i]+1)) for file_name in files_list])>0)[0][0]]
#                     ax1 = fig.add_subplot(1,2,2)
#                     img = mpimg.imread(hmm_plot_dir + '/' + file_name)
#                     ax1.imshow(img)
#                     plt.tight_layout()
#                     fig.savefig(transition_check_dir + '/' + 'on_state%i_trial%i' % (state,orig_trial_index[i]+1))
#                     plt.close(fig)
# =============================================================================
                
                del_transition = del_transition[del_transition!=0]
                transition_trials = np.array([transition[i][0] for i in range(len(transition))])
                ## Now EVERYTHING has to be indexed by transition trials as well -- MAYBE NOT
                fin_on_trials = on_trials[transition_trials]


                trial_palatability = np.zeros(len(palatability_order)*taste_n)
                for i in range(0,len(palatability_order)):
                    trial_palatability[i*taste_n:(i+1)*taste_n] = np.repeat(palatability_order[i],taste_n)[:]
                #trial_palatability = np.array(trial_palatability)

                # Get the spike array for each individual taste and merge all:
                spikes = np.concatenate(tuple(dig_in[i].spike_array[:] for i in range(len(dig_in))), axis = 0)
                on_spikes = spikes[on_trials,:,:]

                # Take spikes from trials with transitions and align them according to state transition per trial
                lined_spikes = []
                for trial, time in transition:
                    lined_spikes.append(on_spikes[trial, :, pre_stim_t + t[time] - data_window_pre :pre_stim_t + t[time] + data_window_post])
                # lined_spikes shape = list of trials, every index = array[neurons , time]

                #for trial in transition[0]:
                #    for time in transition[1]:
                #        lined_spikes.append(spikes[trial, :, 1500 + t[time]:2500 + t[time]])

                lined_spikes = np.array(lined_spikes) # [trials, neurons, time]

                ## Making PSTHs for the spikes
                lined_firing = np.zeros((lined_spikes.shape[0], lined_spikes.shape[1], int((data_window_pre + data_window_post - bin_window_size)/step_size) + 1))

                for i in range(lined_firing.shape[0]):
                    for j in range(lined_firing.shape[1]):
                        for k in range(lined_firing.shape[2]):
                            lined_firing[i, j, k] = np.mean(lined_spikes[i, j, step_size*k:step_size*k + bin_window_size])

###########################################                
                ## Sanity check: at transition, there should be a signficant change in firing pattern
                ## compared to unaligned trials
                
                ## for making raster plots if ever needed 
                #lined_spikes_re = lined_spikes.reshape(lined_spikes.shape[0]*lined_spikes.shape[1],lined_spikes.shape[2])
                #for nrn in range(lined_spikes_re.shape[0]):
                #    x = np.where(lined_spikes_re[nrn,:]==1)
                #    for spike in range(len(x)):
                #        plt.vlines(x,nrn,nrn+1)
                nrn_dec = []
                nrn_inc = []
                #nrn_rank_vec = np.zeros(lined_firing_re.shape[0])
                lined_firing_re = lined_firing.reshape(lined_firing.shape[0]*lined_firing.shape[1],lined_firing.shape[2])
                
                for nrn in range(lined_firing_re.shape[0]):
                    if np.sum(lined_firing_re[nrn,:int(np.floor(lined_firing_re.shape[1]/2))]) > np.sum(lined_firing_re[nrn,int(np.floor(lined_firing_re.shape[1]/2)):]):
                        nrn_dec.append((lined_firing_re[nrn,:]-np.mean(lined_firing_re[nrn,:])))#/np.std(lined_firing_re[nrn,:]))
                    else:
                        nrn_inc.append((lined_firing_re[nrn,:]-np.mean(lined_firing_re[nrn,:])))#/np.std(lined_firing_re[nrn,:]))
                    #first_half = np.sum(lined_firing_re[nrn,:int(np.floor(lined_firing_re.shape[1]/2))])
                    #second_half = np.sum(lined_firing_re[nrn,int(np.floor(lined_firing_re.shape[1]/2)):])
                    #nrn_rank_vec[nrn] = first_half/second_half
                    
                #nrn_rank_vec[np.where(np.isnan(nrn_rank_vec)+np.isinf(nrn_rank_vec))] = 0
                #nrn_ranks = rankdata(nrn_rank_vec,method='ordinal')
                #lined_firing_fin = np.array([lined_firing_re[i,:] for i in nrn_ranks-1])
                #plt.imshow(lined_firing_re)
                
                nrn_dec = np.array(nrn_dec)
                nrn_inc = np.array(nrn_inc)
                all_nrn = np.concatenate((nrn_dec,nrn_inc),axis=0)
                
                fir_plots = plt.figure()
                plt.subplot(121)
                plt.imshow(all_nrn/np.nanmax(all_nrn))
                plt.title('Aligned')
                
                # Unaligned spikes
                mid_point = 900
                on_trans = on_spikes[transition_trials,:]
                on_scram = on_trans.reshape(on_trans.shape[0]*on_trans.shape[1],on_trans.shape[2])
                on_scram_fin = on_scram[:,(pre_stim_t + mid_point - data_window_pre):(pre_stim_t + mid_point + data_window_post)]
                
                on_firing = np.zeros((on_scram_fin.shape[0], int((data_window_pre + data_window_post - bin_window_size)/step_size) + 1))
                

                for i in range(on_firing.shape[0]):
                    for j in range(on_firing.shape[1]):
                            on_firing[i, j] = np.mean(on_scram_fin[i, step_size*j:step_size*j + bin_window_size])
                            
                scram_dec = []
                scram_inc = []
                all_scram_std = np.std(lined_firing_re,axis=1)
                all_scram_std[all_scram_std == 0] = np.mean(all_scram_std)
                for nrn in range(on_firing.shape[0]):
                    if np.sum(on_firing[nrn,:int(np.floor(on_firing.shape[1]/2))]) > np.sum(on_firing[nrn,int(np.floor(on_firing.shape[1]/2)):]):
                        scram_dec.append((on_firing[nrn,:]-np.mean(on_firing[nrn,:])))#/np.std(lined_firing_re[nrn,:]))
                    else:
                        scram_inc.append((on_firing[nrn,:]-np.mean(on_firing[nrn,:])))#/np.std(lined_firing_re[nrn,:]))
                        
                scram_dec = np.array(scram_dec)
                scram_inc = np.array(scram_inc)
                all_scram_nrn = np.concatenate((scram_dec,scram_inc),axis=0)
                
                #plt.figure(2)
                plt.subplot(122)
                plt.imshow(all_scram_nrn/np.nanmax(all_scram_nrn))
                plt.title('Unligned')
                plt.tight_layout()
                
                plt.suptitle('on-firing state %i' % state)
                
                fir_plots.savefig(plot_dir + '/on_firing_rate%i.png' % state)
                plt.close(fir_plots)
                
                ##Cumulative firing plot to go along with the heatmap
                sum_plots = plt.figure()
                plt.subplot(121)
                sum_firing_aligned = np.sum(all_nrn,axis=0)
                plt.plot(sum_firing_aligned)
                plt.title('Aligned')
                plt.subplot(122)
                sum_firing_unaligned = np.sum(all_scram_nrn,axis=0)
                plt.plot(sum_firing_unaligned)
                plt.title('Unaligned')
                plt.suptitle('on-firing cumulative state %i' % state)
                sum_plots.savefig(plot_dir + '/on_firing_rate_sum%i.png' % state)
                plt.close(sum_plots)

###########################################  

                test = lined_firing
                test = np.sum(test,0)
                for i in range(test.shape[0]):
                    test[i,:] = np.log2(test[i,:]/test[i,0])

                fig = plt.figure()
                plt.title('Aligned firing rates, On trials , Model states =' + str(state))
                for i in range(test.shape[0]):
                    plt.plot(np.arange(-data_window_pre,data_window_post-bin_window_size+step_size,step_size),test[i,:])
                plt.xlabel('State transition time')
                plt.ylabel('Normalized Firing Rate')
                fig.savefig(plot_dir + '/on_firing_states%i.png' % state)
                plt.close(fig)


                #palatability = np.ones(lined_firing.shape[0])
                #palatability = np.array([trial_palatability[transition[i][0]] for i in range(0,len(transition))])
                palatability = trial_palatability[fin_on_trials]

                r_pearson = np.zeros((lined_firing.shape[1], lined_firing.shape[2]))
                p_pearson = np.ones((lined_firing.shape[1], lined_firing.shape[2]))

                for i in range(r_pearson.shape[0]):
                    for j in range(r_pearson.shape[1]):
                        r_pearson[i, j], p_pearson[i, j] = pearsonr(lined_firing[:, i, j], palatability)
                        if np.isnan(r_pearson[i, j]):
                            r_pearson[i, j] = 0.0
                            p_pearson[i, j] = 1.0
                            
                pearson_max = np.max(np.mean(r_pearson[:, :]**2, axis = 0))
                pearson_min = np.min(np.mean(r_pearson[:, :]**2, axis = 0))
                pearson_min_max = pearson_max - pearson_min
                fig = plt.figure()
                plt.title('Correlation, On trials , Model states =' + str(state) + ', Trials = ' + str(len(transition)))
                plt.plot(np.arange(-data_window_pre,data_window_post-bin_window_size+step_size,step_size),np.mean(r_pearson[:, :]**2, axis = 0))
                plt.xlabel('State transition time')
                plt.ylabel('Pearson Correlation Coefficient')
                #ax = fig.add_subplot(111)
                #ax.text(-400,pearson_min+(pearson_min_max*0.9),'Mean state transition \n %i +\- %i msec' % (np.mean(del_transition),np.std(del_transition)))
                fig.savefig(plot_dir + '/on_correlation_states%i.png' % state)
                plt.close(fig)


            else:
                r_pearson = []
                p_pearson = []
                del_transition = []
                fig = plt.figure()
                plt.title('Correlation, On trials , Model states =' + str(state) + ', Trials = 0')
                plt.plot(0)
                plt.xlabel('State transition time')
                plt.ylabel('Pearson Correlation Coefficient')
                #ax = fig.add_subplot(111)
                #ax.text(-400,pearson_min+(pearson_min_max*0.9),'Mean state transition \n %i +\- %i msec' % (np.mean(del_transition),np.std(del_transition)))
                fig.savefig(plot_dir + '/on_correlation_states%i.png' % state)
                plt.close(fig)
                
            hf5.create_array('/spike_trains/multinomial_hmm_results/laser_on/states_%i' % state, 'r_pearson', r_pearson)
            hf5.create_array('/spike_trains/multinomial_hmm_results/laser_on/states_%i' % state, 'p_pearson', p_pearson)
            hf5.create_array('/spike_trains/multinomial_hmm_results/laser_on/states_%i' % state, 'palatability_state_transition_t', del_transition)
            hf5.flush()


        hf5.close()
