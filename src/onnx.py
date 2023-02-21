import torch.onnx
import onnx
import onnxruntime as ort
from argparse import ArgumentParser
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from constant import torch_model_path, onnx_based_model, onnx_optimized_model, onnx_quantized_model
from constant import onnx_model_path, onnx_path, dummy_input

def converter(torch_path, onnx_path, onnx_model_path, device, dummy_input):
    # Set model to inference mode, which is required before exporting the model because some operators behave differently in 
    # inference and training mode.
    model = AutoModelForSequenceClassification.from_pretrained(torch_path)
    tokenizer = AutoTokenizer.from_pretrained(torch_path)
    tokenizer.save_pretrained(onnx_path)
    model.eval()
    model.to(device)
    max_seq_length = model.config.max_position_embeddings
    
    # Create the dummy input tensor
    dummy_inputs = tokenizer(dummy_input, return_tensors='pt', truncation=True,
                             padding='max_length', max_length=max_seq_length)
    inputs = {
        'input_ids': dummy_inputs['input_ids'].to(device).reshape(1, max_seq_length),
        'attention_mask': dummy_inputs['attention_mask'].to(device).reshape(1, max_seq_length)
    }

    # Export model to ONNX 
    symbolic_names = {0: 'batch_size', 1: 'max_seq_length'}
    torch.onnx.export(model,    #model being run
                      (inputs['input_ids'],
                       inputs['attention_mask']), # model input (or a tuple for multiple inputs)
                       onnx_model_path, # where to save the model (can be a file or file-like object)
                       opset_version=14, # the ONNX version to export the model to
                       do_constant_folding=False,  # whether to execute constant folding for optimization
                       input_names=['input_ids', 'attention_mask'],  # the model's input names
                       output_names=['logits'],  # the model's output names
                       dynamic_axes ={'input_ids': symbolic_names,
                                      'attention_mask': symbolic_names,
                                      'logits': symbolic_names},
                       verbose=True)
    print('Model has been converted to ONNX at', onnx_model_path)

def checker(onnx_model_path):
    try:
        onnx.checker.check_model(onnx_model_path)
    except onnx.checker.ValidationError as e:
        print("The model is not consistent: %s" % e)
    else:
        print("The model is valid!")

def graph_optimizer(onnx_based_model, onnx_optimzed_model):
    sess_options = ort.SessionOptions()
    # Set graph optimization level
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    # To enable model serialization after graph optimization set this
    sess_options.optimized_model_filepath = onnx_optimzed_model
    session = ort.InferenceSession(onnx_based_model, sess_options)

def quantizer_dynamic(onnx_based_model, onnx_quantized_model):
    model_input = onnx_based_model
    model_quant = onnx_quantized_model
    quantized_model = quantize_dynamic(model_input, model_quant, weight_type=QuantType.QInt8)
    return quantized_model

def cli():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    convert = subparsers.add_parser("convert")
    check = subparsers.add_parser("check")
    optimize = subparsers.add_parser("optimize")
    quantize = subparsers.add_parser("quantize")

    convert.add_argument('--torch-path', '-tp', default=torch_model_path, type=str,
                         help='Provide based model in torch format.')
    convert.add_argument('--onnx-path', '-oxp', default=onnx_path, type=str,
                         help='Path to save onnx models & tokenier.')
    convert.add_argument('-onnx-model-path', '-oxm', default=onnx_based_model, type=str,
                         help='Full path name contains model name. Example: onnx/onnx_optimized.onnx')
    convert.add_argument('--device', '-d', default=torch.device("cpu"),
                         help='Device onnx model will run on.')
    convert.add_argument('--dummy-input', '-dummy', default=dummy_input, type=str,
                         help='Add an example of input.')
    
    check.add_argument('--onnx-model-path', default=onnx_model_path, type=str,
                       help='Full onnx model path to check consistency.')
    
    optimize.add_argument('--onnx-based-model', default=onnx_based_model, type=str,
                          help='Full path name of onnx model that need to optimize.')
    optimize.add_argument('--onnx_optimized_model', default=onnx_optimized_model, type=str,
                          help='Full path name contains model name. Example: onnx/onnx_optimized.onnx')
    
    quantize.add_argument('--onnx-based-model', default=onnx_based_model, type=str,
                          help='Full path name of onnx model that need to quantize.')
    quantize.add_argument('--onnx_quantized_model', default=onnx_quantized_model, type=str,
                          help='Full path name contains model name. Example: onnx/onnx_quantized.onnx')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = cli()
    if args.command=='convert':
        converter(args)
    elif args.command=='check':
        checker(args)
    elif args.command=='optimize':
        graph_optimizer(args)
    elif args.command=='quantize':
        quantizer_dynamic(args)
    else:
        print('Please choose a compatible command.')
