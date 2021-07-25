for line in $(cat split_files.txt);
do
    figlet ==
    echo $line
    python correlation_hannah_updated.py $line;
done
