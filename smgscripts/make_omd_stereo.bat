python -u inference.py --task=stereo --data_root="D:/datasets" --data_list="smglists/omd_stereo_%1_%2.txt" --context --encoder=dlaup --decoder=hda --batch_size=1 --workers=16 --flow_format=png --model_path="model_zoo/hd3sc_things_kitti-368975c0.pth" --save_folder="D:/datasets/hd3/omd_stereo"