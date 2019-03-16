#!/bin/bash

ret=0
i=0
while [ $ret -eq 0 ]; do
    echo "start experiment $i"
    python experiment.py -id $i >$i.out >>$i.err
    ret=$?
    ((i++))
done
