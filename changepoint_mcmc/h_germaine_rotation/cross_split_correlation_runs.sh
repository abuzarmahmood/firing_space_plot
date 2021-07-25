for line in $(cat split_analyses_folders.txt);
do
    figlet ==
    echo $line
    python cross_split_correlation.py $line;
done
