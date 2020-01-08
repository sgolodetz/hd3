python -u inference.py \
  --task=flow \
  --data_root=/media/data/datasets \
  --data_list=smglists/omd_flow_$1_$2.txt \
  --context \
  --encoder=dlaup \
  --decoder=hda \
  --batch_size=1 \
  --workers=16 \
  --flow_format=flo \
  --model_path=model_zoo/hd3fc_chairs_things_kitti-bfa97911.pth \
  --save_folder=/media/data/datasets/hd3/omd_flow2d
