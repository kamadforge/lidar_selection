#!/bin/bash

files=(2011_09_26_calib.zip
2011_09_28_calib.zip
2011_09_29_calib.zip
2011_09_30_calib.zip
2011_10_03_calib.zip)

for i in ${files[@]}; do
        if [ ${i:(-3)} != "zip" ]
        then
                shortname=$i'_sync.zip'
                fullname=$i'/'$i'_sync.zip'
        else
                shortname=$i
                fullname=$i
        fi
	echo "Downloading: "$shortname
        wget 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'$fullname
        unzip -o $shortname
        rm $shortname
done
