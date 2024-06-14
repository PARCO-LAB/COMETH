#!/bin/bash

python3 to_viewer.py tmp/$1.json  \
        --rotation 180 --output "tmp/"$1".viewer.json" >> "tmp/"$1"_log_2.txt" \
