for line in $(cat regression_file_list.txt);do figlet _; 
    echo $line; figlet _;
    python inter_region_regression_test.py $line;
done
