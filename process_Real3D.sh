# Convert Real3D to .tiff
python Real2Depth.py


data_dir=../../datasets/Real3D
save_dir=../../datasets/Real3D-multiview
export DISPLAY=:0
cd utils

python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category airplane --save-dir $save_dir
python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category candybar --save-dir $save_dir
python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category car --save-dir $save_dir
python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category chicken --save-dir $save_dir
python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category diamond --save-dir $save_dir
python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category duck --save-dir $save_dir
python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category fish --save-dir $save_dir
python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category gemstone --save-dir $save_dir
python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category seahorse --save-dir $save_dir
python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category shell --save-dir $save_dir
python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category starfish --save-dir $save_dir
python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category toffees --save-dir $save_dir
cd ..