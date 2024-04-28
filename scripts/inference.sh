train_mode=1
inference_dataset='./data/inference.csv'
save_path='./save'
preference_num=16
batch_size=64

python main.py --train_mode $train_mode --test_dataset $inference_dataset --model_save $save_path --load_file $uberl_ep99 --preference_num $preference_num --batch_size $batch_size