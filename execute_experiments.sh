#!/bin/bash

ret=0
i=0
folder=network_metadata
mkdir $folder
while [ $ret -eq 0 ]; do
    echo "start experiment $i"
    python experiment.py -id $i >$folder/$i.out 2>$folder/$i.err
    ret=$?
    ((i++))
done
rm $folder/$(($i-1)).*
for a in $(seq 0 $(($i-2))); do
    file_name=$(grep -m 1 data $folder/$a.out)
    if [[ ! -z "$file_name" ]]; then
        mv $folder/$a.out $folder/$file_name
        mv $folder/$a.err $folder/$file_name.err
    fi
done

cd memory_usage
python parse.py
