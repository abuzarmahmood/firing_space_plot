#data_super_dir='/media/bigdata/Abuzar_Data'  
#file_list_dir='/media/bigdata/firing_space_plot/changepoint_mcmc/file_lists'
#find $data_super_dir -iname "*pkl" > $file_list_dir/fitted_models.txt
#cat $file_list_dir/fitted_models.txt | grep -v "shuffle\|simulate" | grep "4state\|5state" | sort > $file_list_dir/actual_data_models.txt
#for line in $(cat $file_list_dir/actual_data_models.txt);

file_list_dir='/media/bigdata/firing_space_plot/changepoint_mcmc/file_lists/'
file=$1

grep -oP '.+\dTastes_\d+_\d+' $file | sort | uniq > $file_list_dir/analysis_unique_dirs.txt

serial_analysis() {
    input=$1
    for line in $input
    do 
        figlet ==
        echo $(dirname $line);
        python changepoint_analysis.py $line; 
    done
}

num_processes=12
for line in $(cat $file_list_dir/analysis_unique_dirs.txt)
do
    ((i=i%num_processes)); ((i++==0)) && wait
    echo $line
    serial_analysis "$(grep $line $file)" &
done

#file_list=$1
#for line in $(cat $file_list) 
#    do figlet ==
#    echo $(dirname $line);
#    python changepoint_analysis.py $line; 
#done
