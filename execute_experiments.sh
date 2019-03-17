#!/bin/bash

number_regex='^[0-9]+$'
folder=network_metadata
mkdir $folder

function run_experiment {
    echo "run experiment $2"
    /usr/bin/time -lp python experiment.py $1 $2 >$folder/$3.out 2>$folder/$3.err
    return $?
}

function rename_file {
    file_name=$1
    new_file_name=$(grep -m 1 data_ $folder/$file_name.out)
    if [[ ! -z "$new_file_name" ]]; then
        mv $folder/$file_name.out $folder/_$new_file_name
        mv $folder/$file_name.err $folder/$new_file_name.err
    fi
}

i=0
if [[ ! -z "$1" ]]; then
    echo "executing chosen experiments"
    for test_id in $@; do
        if [[ $test_id =~ $number_regex ]]; then
            run_experiment -id $test_id $i
        else
            run_experiment -name $test_id $i
        fi
        ((i++))
    done
else
    echo "executing all experiments"
    ret=0
    while [ $ret -eq 0 ]; do
        run_experiment -id $i $i
        ret=$i
        ((i++))
    done
    ((i--))
    rm $folder/$i.*
fi

for file_name in $(seq 0 $(($i-1))); do
    rename_file $file_name
done

cd memory_usage
python parse.py
