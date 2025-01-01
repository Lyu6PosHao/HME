#model_args
model=path/to/HME/checkpoint  #HME checkpoint's path 
max_length=512                #max_length of HME tokenization

#data_args
data_path=path/to/test_dataset/of/task.json  #path to json file of the test dataset
data_type=1d,2d,3d,frg     #1d, 2d, 3d, frg: see data.py for more details
task_type=caption   #captioning, see data.py for more details


#other_args
save_specify=a_name_for_save_file   #specify the name of the output file
number=0                      #GPU number
port=$((number + 29504))      #port number for torchrun
output_path=path/to/save/file/${save_specify}_${task_type}.txt  #where to save the output file


CUDA_VISIBLE_DEVICES=$((number)) torchrun --nnodes 1 --nproc_per_node 1 --master_port $port infer.py \
    --model_name_or_path $model \
    --max_length $max_length \
    --data_path $data_path \
    --data_type $data_type \
    --task_type $task_type \
    --output_path $output_path 