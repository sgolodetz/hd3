python -u inference.py --task=flow --data_root="D:/datasets" --data_list="smglists/KITTI_raw_rflow_%1.txt" --context --encoder=dlaup --decoder=hda --batch_size=1 --workers=16 --flow_format=flo --model_path="model_zoo/hd3fc_chairs_things_kitti-bfa97911.pth" --save_folder="D:/datasets/hd3/kitti_raw_rflow2d"