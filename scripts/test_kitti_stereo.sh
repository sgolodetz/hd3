python -u inference.py \
  --task=stereo \
  --data_root=/media/data/datasets \
  --data_list=lists/KITTI_stereo_test_2015.txt \
  --context \
  --encoder=dlaup \
  --decoder=hda \
  --batch_size=1 \
  --workers=16 \
  --flow_format=png \
  --model_path=model_zoo/hd3sc_things_kitti-368975c0.pth \
  --save_folder=kitti_stereo