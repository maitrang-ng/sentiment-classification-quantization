# Define constant variables
pretrained_model = 'distilbert-base-uncased'
torch_model_path = '../models/DistilBert'
onnx_model_path = '../models/ONNX/distilbert_full_precision.onnx'
onnx_path = '../models/ONNX'

onnx_based_model = '../models/ONNX/vanilla.onnx'
onnx_optimized_model = '../models/ONNX/graph_optimized.onnx'
onnx_quantized_model = '../models/ONNX/quantized.onnx'

# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)
# max_seq_length = model.config.max_position_embeddings
max_seq_length = 512

train_dataset = '../dataset/train.csv'
test_dataset = '../dataset/test.csv'
dummy_input = str("It doesn\'t matter what one\'s political views are because this film can hardly be taken seriously on any level. As for the claim that frontal male nudity is an automatic NC-17, that isn\'t true. I\'ve seen R-rated films with male nudity. Granted, they only offer some fleeting views, but where are the R-rated films with gaping vulvas and flapping labia? Nowhere, because they don\'t exist. The same goes for those crappy cable shows: schlongs swinging in the breeze but not a clitoris in sight.")