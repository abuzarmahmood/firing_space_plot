for line in $(cat *.txt);do figlet _; echo $(dirname $line); figlet _;
    python inter_region_noise_corrs_setup.py $(dirname $line);
    python inter_region_noise_corrs_plot.py $(dirname $line);
done
