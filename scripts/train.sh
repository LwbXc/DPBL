pretrain_dataset='./data/pretrain.csv'
save_path='./save'
model_name='uberl'
preference_num=16
epochs=100
batch_size=64

python main.py --train_dataset $pretrain_dataset --model_save $save_path --model_name $model_name --preference_num $preference_num --epochs $epochs --batch_size $batch_size