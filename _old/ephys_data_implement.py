import numpy as np
from ephys_data import ephys_data
test = ephys_data(data_dir = '/media/bigdata/NM_2500/file_1/',file_id = 1)
test.get_data()
test.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                                   [25,250,7000]))
test.get_firing_rates()
test.correlation_params = dict(zip(['stimulus_start_time', 'stimulus_end_time', 
            'baseline_start_time', 'baseline_end_time',
            'baseline_window_sizes', 'shuffle_repeats'], 
            [2000, 4000, 200, 2000, np.arange(100,1000,100), 100]))
test.get_baseline_windows()
test.get_correlations()
