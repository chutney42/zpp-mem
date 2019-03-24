#!/bin/bash

number_regex='^[0-9]+$'
folder=network_metadata
mkdir ${folder}

# runs single experiment with parameters:
# $1 - type of network identifier (either "-id" or "-name")
# $2 - network identifier (e.g. "default_network" or "1")
# $3 - prefix of output files (e.g. "0")
# returns return code of experiment
# stores stdout and stderr in $3.out and $3.err in $folder catalogue
function run_experiment {
    echo "run experiment $2"
    /usr/bin/time -lp python experiment.py $1 $2 >${folder}/$3.out 2>${folder}/$3.err
    return $?
}

# renames stdout and stderr files of a run with specified prefix:
# $1 - prefix of files to rename (e.g. "0")
# stdout file is renamed to _$a
# stderr file is renamed to $a.err
# where $a is a run name collected from stdout file (e.g. "data_BP_12")
function rename_file {
    file_name=$1
    new_file_name=$(grep -m 1 data_ ${folder}/${file_name}.out)
    if [[ ! -z "$new_file_name" ]]; then
        mv ${folder}/${file_name}.out ${folder}/_${new_file_name}
        mv ${folder}/${file_name}.err ${folder}/${new_file_name}.err
    fi
}

iterator=0
# if script is called with parameters, run every requested experiment
if [[ ! -z "$1" ]]; then
    echo "executing chosen experiments"
    for test_id in $@; do
        if [[ ${test_id} =~ $number_regex ]]; then
            run_experiment -id ${test_id} ${iterator}
        else
            run_experiment -name ${test_id} ${iterator}
        fi
        ((iterator++))
    done
# else run all experiments (run experiment with each id starting from 0,
# and stop when run_experiment returns error code)
else
    echo "executing all experiments"
    ret=0
    while [[ ${ret} -eq 0 ]]; do
        run_experiment -id ${iterator} ${iterator}
        ret=$?
        ((iterator++))
    done
    ((iterator--))
    rm ${folder}/${iterator}.*
fi

# rename all stdout and stderr files
for file_name in $(seq 0 $(($iterator-1))); do
    rename_file ${file_name}
done

# add memory metadata to stdout files,
# and remove first underscore from that files' names (if succeed)
cd memory_usage
python parse.py
