#! /bin/bash -e

rm -f flow*.txt

ls /media/data/datasets/kitti_raw/$1/$2/image_02/data | while read f
do
	echo "kitti_raw/$1/$2/image_02/data/$f" >> flow1.txt
	cp flow1.txt flow2.txt
	head -n -1 flow1.txt > flow3.txt
	tail -n +2 flow2.txt > flow4.txt
	paste -d ' ' flow4.txt flow3.txt > "KITTI_raw_rflow_$2.txt"
done

rm -f flow*.txt

