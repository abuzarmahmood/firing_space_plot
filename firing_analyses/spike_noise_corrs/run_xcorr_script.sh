for line in $(cat good*paths.txt);do figlet _; echo $line; figlet _;
    #python inter_region_rolling_noise_corrs_setup.py $line;
    python inter_region_rolling_noise_corrs_plots.py $line;
    #python inter_region_noise_corrs_plot.py $line;
done
