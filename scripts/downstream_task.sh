training_emmbedding_path='./data/train_embedding.json'
test_embedding_path='./data/test_embedding.json'
train_labels='./data/train_labels.json'
test_labels='./data/test_labels.json'
epochs=100
log_path='./logs/log.txt'

python classifier/classifier.py --train_embeddings $training_emmbedding_path --test_embeddings $test_embedding_path --train_labels $train_labels --test_labels $test_labels --epochs $epochs --result_save $log_path