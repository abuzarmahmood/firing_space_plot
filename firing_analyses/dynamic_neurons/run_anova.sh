FILE_LIST=$1
for line in $(cat $FILE_LIST);do figlet _; echo $(basename $line); figlet _;
    python time_bin_anova.py $line
done
