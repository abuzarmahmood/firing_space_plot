"""
Example run for changepoint fit
"""
import sys
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
sys.path.append('/media/bigdata/firing_space_plot/changepoint_mcmc/v2')
from ephys_data import ephys_data
from changepoint_io import fit_handler

data_dir = '/media/bigdata/Abuzar_Data/bla_gc/AM11/AM11_4Tastes_191029_171714'
taste_num = 0
region_name = 'bla'
experiment_name = 'single_taste_poisson'

model_parameters = dict(zip(['states','fit','samples'],[4,4000,2000]))
preprocess_parameters = dict(zip(['time_lims','bin_width','data_transform'],
                                [[2000,4000],50, None]))

fit_handler_kwargs = {'data_dir' : data_dir,
                    'taste_num' : taste_num,
                    'region_name' : region_name,
                    'experiment_name' : experiment_name}
handler = fit_handler(**fit_handler_kwargs)
handler.set_model_params(**model_parameters)
handler.set_preprocess_params(**preprocess_parameters)
handler.run_inference()
handler.save_fit_output()
