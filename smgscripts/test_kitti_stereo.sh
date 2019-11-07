python -u inference.py \
  --task=stereo \
  --data_root=/media/data/datasets \
  --data_list=smglists/KITTI_raw_stereo_2011_09_26_drive_0005_sync.txt \
  --context \
  --encoder=dlaup \
  --decoder=hda \
  --batch_size=1 \
  --workers=16 \
  --flow_format=png \
  --model_path=model_zoo/hd3sc_things_kitti-368975c0.pth \
  --save_folder=/media/data/datasets/hd3_prob/kitti_raw_stereo
