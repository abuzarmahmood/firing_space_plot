file_list=$1
#states=$2
for states in {7,8,9,10}
do
    for line in $(cat $file_list);
    do 
        echo $(basename $line) $states;
        python changepoint_fit.py $line $states;
    done
done
