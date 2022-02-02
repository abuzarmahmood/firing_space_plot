for line in $(cat *.txt);do figlet _; echo $line; figlet _;
    #python lfp_power_rolling_xcorr_setup.py $line;
    python lfp_power_rolling_xcorr_plot.py $line;
    #python lfp_power_xcorr_plot.py $(dirname $line);
done
