import sys
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
sys.path.append('/media/bigdata/firing_space_plot/changepoint_mcmc/v2')
from ephys_data import ephys_data
from changepoint_io import fit_handler
import itertools as it
from tqdm import tqdm
import os

dir_list_path = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'transition_corrs/all_tastes/inter_region_dirs.txt'
dir_list = [x.strip() for x in open(dir_list_path,'r').readlines()]
taste_num_list = range(3)
region_list = ['bla','gc']
preprocess_list = ['shuffled','simulated']

arg_list = list(it.product(dir_list,taste_num_list,region_list, preprocess_list))

for data_dir, taste_num, region_name, preprocess_transform in tqdm(arg_list):
    #data_dir = '/media/bigdata/Abuzar_Data/bla_gc/AM11/AM11_4Tastes_191029_171714'
    #taste_num = 0
    #region_name = 'bla'
    experiment_name = 'single_taste_poisson'

    model_parameters = dict(zip(['states','fit','samples'],[4,40000,20000]))
    preprocess_parameters = dict(zip(['time_lims','bin_width','data_transform'],
                                    [[2000,4000],50, preprocess_transform]))

    fit_handler_kwargs = {'data_dir' : data_dir,
                        'taste_num' : taste_num,
                        'region_name' : region_name,
                        'experiment_name' : experiment_name}
    handler = fit_handler(**fit_handler_kwargs)
    handler.set_model_params(**model_parameters)
    handler.set_preprocess_params(**preprocess_parameters)

    error_file_path = os.path.join(
            handler.database_handler.model_save_dir,
            'error_log_file.txt')

    try:
        handler.run_inference()
        handler.save_fit_output()
    except:
        with open(error_file_path,'a') as error_file:
            error_file.write(\
                str((data_dir,taste_num,region_name, preprocess_transform)) + "\n")
