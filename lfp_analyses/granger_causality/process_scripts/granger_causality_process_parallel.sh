dir_list_path=/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt
log_path=/media/bigdata/firing_space_plot/lfp_analyses/granger_causality/granger_log.txt
rm $log_path
for DIR in $(cat $dir_list_path)
do
    echo Processing $DIR
    {
        python granger_causality_process_single.py $DIR &&
        echo Finished $DIR
    } || echo Failed $DIR >> $log_path
done
