#! /bin/bash -e

rm -f stereo*.txt

ls /media/data/datasets/omd/$1/$2/stereo/*left.png | xargs -n1 basename | while read f
do
	echo "omd/$1/$2/stereo/$f" >> stereo1.txt
done

(cat stereo1.txt | perl -pe 's/left/right/g') > stereo2.txt
paste -d ' ' stereo1.txt stereo2.txt > "omd_stereo_$1_$2.txt"
rm -f stereo*.txt

