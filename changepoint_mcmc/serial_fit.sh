file_list=$1
good_bool=$2
simulate_bool=$3
#states=$2
for states in {4}
do
    for line in $(cat $file_list);
    do 
        echo $(basename $line) $states;
        python changepoint_fit.py $line $states $good_bool $simulate_bool;
    done
done
