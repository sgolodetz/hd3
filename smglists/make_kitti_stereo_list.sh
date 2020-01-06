#! /bin/bash -e

rm -f stereo*.txt

ls /media/data/datasets/kitti_raw/$1/$2/image_02/data | while read f
do
	echo "kitti_raw/$1/$2/image_02/data/$f" >> stereo1.txt
	(cat stereo1.txt | perl -pe 's/image_02/image_03/g') > stereo2.txt
	paste -d ' ' stereo1.txt stereo2.txt > "KITTI_raw_stereo_$2.txt"
done

rm -f stereo*.txt

