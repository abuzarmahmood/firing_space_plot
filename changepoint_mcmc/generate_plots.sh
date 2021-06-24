data_super_dir='/media/bigdata/Abuzar_Data'  
find $data_super_dir -iname "*pkl" > fitted_models.txt
for line in $(cat fitted_models.txt);
do
    figlet ==
    echo $line
    python changepoint_plots.py $line;
done
