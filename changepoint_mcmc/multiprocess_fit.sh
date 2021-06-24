#run_fits() {
#    dir_name=$1
#    states=$2
#    figlet --
#    echo $(basename $dir_name);
#    python changepoint_fit.py $dir_name $states
#    #python changepoint_shuffle.py $dir_name $states
#}

# Multiprocess each serial fit command so that multiple processes aren't
# trying to access the same HDF5 file
serial_fit() {
    line=$1
    good_bool=$2
    simulate_bool=$3
    for states in {4,}
    do 
        figlet ==
        echo $(basename $line) $states;
        python changepoint_fit.py $line $states --good $good_bool --simulate $simulate_bool;
    done
}

num_processes=12
#states=$2
file_list=$1
good_bool=$2
simulate_bool=$3
for line in $(cat $file_list)
do 
    ((i=i%num_processes)); ((i++==0)) && wait
    #run_fits "$line" "$states" &
    serial_fit "$line" "$good_bool" "$simlate_bool" &
done
