#! /bin/bash -e

rm -f flow*.txt

ls /media/data/datasets/omd/$1/$2/stereo/*left.png | xargs -n1 basename | while read f
do
	echo "omd/$1/$2/stereo/$f" >> flow1.txt
	cp flow1.txt flow2.txt
	head -n -1 flow1.txt > flow3.txt
	tail -n +2 flow2.txt > flow4.txt
	paste -d ' ' flow4.txt flow3.txt > "omd_rflow_$1_$2.txt"
done

rm -f flow*.txt
