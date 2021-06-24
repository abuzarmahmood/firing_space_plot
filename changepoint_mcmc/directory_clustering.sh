file_list_dir='/media/bigdata/firing_space_plot/changepoint_mcmc/file_lists/'
file=$file_list_dir/actual_data_models.txt
grep -oP '.+\dTastes_\d+_\d+' $file | sort | uniq > $file_list_dir/actual_data_models_unique_dirs.txt 

echo_test() {
    input=$1
    for line in $input
    do
        echo $line
        echo 'break here'
    done
}

#serial_plotting() {
#    input=$1
#    for line in $input
#    do 
#        figlet ==
#        echo $(dirname $line);
#        python changepoint_analysis.py $line $1; 
#    done
#}

for line in $(cat $file_list_dir/actual_data_models_unique_dirs.txt)
do
    echo $line
    echo_test "$(grep $line $file)"
done
