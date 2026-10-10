#!/bin/bash

while true
do
    for i in {1..19}
    do
        hyprctl notify 5 4000 "rgb(88ffaa)" "fontsize:99 TRAINING FINISHED" &> /dev/null
        sleep 0.025
    done
    # sleep 3.33
    sleep 4
done
