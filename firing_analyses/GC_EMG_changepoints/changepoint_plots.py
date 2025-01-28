## Import modules
base_dir = '/media/bigdata/projects/pytau'
import sys
sys.path.append(base_dir)
from pytau.changepoint_io import FitHandler
import pylab as plt
from pytau.utils import plotting
from pytau.utils import ephys_data
from tqdm import tqdm
from pytau.changepoint_io import DatabaseHandler
from pytau.changepoint_analysis import PklHandler, get_transition_snips
import os
import pandas as pd

# fit_database = DatabaseHandler()
# fit_database.drop_duplicates()
# fit_database.clear_mismatched_paths()
# 
# # Get fits for a particular experiment
# dframe = fit_database.fit_database
# wanted_exp_name = 'GC_EMG_changepoints_single_taste'
# wanted_frame = dframe.loc[dframe['exp.exp_name'] == wanted_exp_name] 
# # Pull out a single data_directory

data_dir_file = '/media/bigdata/firing_space_plot/firing_analyses/GC_EMG_changepoints/data_dir_list.txt'
with open(data_dir_file, 'r') as f:
    data_dir_list = f.read().splitlines()

base_plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/GC_EMG_changepoints/plots'
change_plot_dir = os.path.join(base_plot_dir, 'changepoint_plots')

artifact_dir = '/media/bigdata/firing_space_plot/firing_analyses/GC_EMG_changepoints/artifacts'

scaled_mode_tau_df = pd.read_pickle(os.path.join(artifact_dir, 'scaled_mode_tau_cut.pkl'))

transition_snips_list = []
# for i, this_row in tqdm(wanted_frame.iterrows()):
for i, this_row in tqdm(scaled_mode_tau_df.iterrows()):

    # # i = 0
    # # this_row = wanted_frame.iloc[i]
    # pkl_path = this_row['exp.save_path']
    # basename = this_row['data.basename']
    # taste_num = this_row['data.taste_num']

    # # From saved pkl file
    # this_handler = PklHandler(pkl_path)

    # spike_train = this_handler.firing.raw_spikes
    # scaled_mode_tau = this_handler.tau.scaled_mode_tau

    # From saved dataframe
    basename = this_row['basename']
    # taste_num = int(this_row['base_taste'])
    taste_num = int(this_row['taste'])
    scaled_mode_tau = this_row['scaled_mode_tau']
    data_dir = [x for x in data_dir_list if basename in x][0]
    section_array = this_row['section_array']

    dat = ephys_data.EphysData(data_dir)
    dat.get_spikes()
    spike_train = dat.spikes[taste_num]


    # Get and save transition snips
    this_snips = get_transition_snips(spike_train, scaled_mode_tau, window_radius=300)
    snips_dict = dict(
            basename=basename,
            taste_num=taste_num,
            snips=this_snips
            )
    transition_snips_list.append(snips_dict)

    # Changepoint raster
    fig, ax = plotting.plot_changepoint_raster(
            spike_train, scaled_mode_tau, [2000, 4000],
            figsize = (7, 15)
            )
    n_sections = len(np.unique(section_array))
    cmap = plt.cm.get_cmap('tab20', n_sections)
    colors = [cmap(int(i)) for i in section_array]
    for i in range(len(section_array)):
        ax[i].set_ylabel(f'Sec \n {int(section_array[i])}')
        # Make label bold, and change color by section
        ax[i].set_title(ax[i].get_title(), fontweight='bold')
        ax[i].spines['left'].set_color(colors[i])
        ax[i].spines['left'].set_linewidth(10)
    fig.suptitle(f'{basename} Taste {taste_num}')
    # fig.savefig(os.path.join(change_plot_dir, f'{basename}_taste_{taste_num}_raster.png'))
    fig.savefig(os.path.join(change_plot_dir, f'{basename}_taste_{taste_num}_raster_cut.png'))
    plt.close(fig)

    # Changepoint overview
    fig, ax = plotting.plot_changepoint_overview(
            scaled_mode_tau, [2000, 4000]
            )
    # ax = 2 rows, 1 column
    # Add a thin ax next to the top one to plot the section_array
    # Get edges of top ax
    top_ax = ax[0]
    top_ax_edges = top_ax.get_position().bounds
    # Add vertical thin ax to right of top ax
    sec_ax = fig.add_axes([top_ax_edges[0] + top_ax_edges[2] + 0.01, top_ax_edges[1], 0.01, top_ax_edges[3]],
                          sharey=top_ax)
    sec_ax.imshow(section_array[:, None], aspect='auto', cmap='tab20', origin='lower')
    fig.suptitle(f'{basename} Taste {taste_num}')
    # fig.savefig(os.path.join(change_plot_dir, f'{basename}_taste_{taste_num}_overview.png'))
    fig.savefig(os.path.join(change_plot_dir, f'{basename}_taste_{taste_num}_overview_cut.png'))
    plt.close(fig)

    # Changepoint aligned rasters 
    fig, ax = plotting.plot_aligned_state_firing(
            spike_train, scaled_mode_tau, window_radius=300
            )
    fig.suptitle(f'{basename} Taste {taste_num}')
    # fig.savefig(os.path.join(change_plot_dir, f'{basename}_taste_{taste_num}_aligned_raster.png'))
    fig.savefig(os.path.join(change_plot_dir, f'{basename}_taste_{taste_num}_aligned_raster_cut.png'))
    plt.close(fig)

    # fig, ax = plotting.plot_aligned_state_firing_raster(
    #         spike_train, scaled_mode_tau, window_radius=300
    #         )
    # fig.suptitle(f'{basename} Taste {taste_num}')
    # # fig.savefig(os.path.join(change_plot_dir, f'{basename}_taste_{taste_num}_aligned_raster.png'))
    # fig.savefig(os.path.join(change_plot_dir, f'{basename}_taste_{taste_num}_aligned_raster_cut.png'))
    # plt.close(fig)

    # # Changpoint aligned firing rates
    # fig, ax = plotting.plot_aligned_state_firing_line(
    #         spike_train, scaled_mode_tau, window_radius=300
    #         )
    # fig.suptitle(f'{basename} Taste {taste_num}')
    # # fig.savefig(os.path.join(change_plot_dir, f'{basename}_taste_{taste_num}_aligned_rates.png'))
    # fig.savefig(os.path.join(change_plot_dir, f'{basename}_taste_{taste_num}_aligned_rates_cut.png'))
    # plt.close(fig)

    # State firing rates
    fig, ax = plotting.plot_state_firing_rates(
            spike_train, scaled_mode_tau
            )
    fig.suptitle(f'{basename} Taste {taste_num}')
    # fig.savefig(os.path.join(change_plot_dir, f'{basename}_taste_{taste_num}_rates.png'))
    fig.savefig(os.path.join(change_plot_dir, f'{basename}_taste_{taste_num}_rates_cut.png'))
    plt.close(fig)

    # # State firing rates
    # fig, ax = plotting.plot_state_firing_rates_2(
    #         spike_train, scaled_mode_tau
    #         )
    # plt.tight_layout()
    # fig.suptitle(f'{basename} Taste {taste_num}')
    # # fig.savefig(os.path.join(change_plot_dir, f'{basename}_taste_{taste_num}_rates_2.png'))
    # fig.savefig(os.path.join(change_plot_dir, f'{basename}_taste_{taste_num}_rates_2_cut.png'))
    # plt.close(fig)

    # # State firing rates
    # fig, ax = plotting.plot_state_firing_overlay(
    #         spike_train, scaled_mode_tau
    #         )
    # fig.suptitle(f'{basename} Taste {taste_num}')
    # plt.tight_layout()
    # # fig.savefig(os.path.join(change_plot_dir, f'{basename}_taste_{taste_num}_rates_3.png'))
    # fig.savefig(os.path.join(change_plot_dir, f'{basename}_taste_{taste_num}_rates_3_cut.png'))
    # plt.close(fig)

snips_frame = pd.DataFrame(transition_snips_list)
snips_frame.to_pickle(os.path.join(artifact_dir, 'transition_snips_cut.pkl'))
