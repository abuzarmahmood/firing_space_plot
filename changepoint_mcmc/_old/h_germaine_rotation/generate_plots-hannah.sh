for line in $(cat gc_only_fitted_models1.txt);
do
    figlet ==
    echo $line
    python changepoint_plots-hannah.py $line True;
done
