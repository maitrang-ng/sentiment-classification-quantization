import os
#import sentencepiece
from argparse import ArgumentParser
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from process_data import Dataset
from utils import compute_metrics
from constant import pretrained_model, train_dataset, test_dataset
from process_data import process

def cli():
    parser = ArgumentParser()
    #subparsers = parser.add_subparsers(dest='command')
    #data = subparsers.add_parser("data")
    #train = subparsers.add_parser("train")

    # Arguments users used when running command lines
    parser.add_argument("--dataset", "-ds", nargs="+", type=str, default=[train_dataset, test_dataset],
                        help='Directory of dataset files in order: train, validation, test as a list. Value in list must be string.')
    parser.add_argument("--ratio", "-r", nargs="+", type=float, default=[0.2],                  
                        help='ratio is used in the case you want concatenat all datasets and then split with your own ratio.')    
    
    parser.add_argument("--pretrained-model", "-m", default=pretrained_model, type=str,
                       help='Choose a pre-trained model for fine-tuning.')  
    parser.add_argument("--batch-size", "-b", default=16, type=int)
    parser.add_argument("--epochs", "-e", default=2, type=int)
    parser.add_argument("--learning-rate", "-lr", default=2e-5, type=float)
    parser.add_argument("--weight-decay", "-wd", default=0.01, type=float)
    parser.add_argument("--output-dir", "-o", default='../models/output', type=str)
    parser.add_argument("--save-model-dir", "-s", default='../models/torch', type=str)

    args = parser.parse_args()
    return args

# Set hyperparameters
def main(args, datasets):
    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=2, ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    # Tokenizer inputs
    def tokenization(example):
        return tokenizer(example["text"], padding=True, truncation=True)

    #logging_steps = len(tokenized_train_dataset) // args.batch_size
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay = args.weight_decay,
        evaluation_strategy='steps',
        #logging_steps=logging_steps,
        log_level="error",
        disable_tqdm=False,
        load_best_model_at_end=True,
        save_total_limit=2)

    if len(datasets)==3:
        trainer = Trainer(
            model=model,
            args=training_args, 
            train_dataset=datasets[0].map(tokenization, batched=True),
            eval_dataset=datasets[1].map(tokenization, batched=True),
            compute_metrics=compute_metrics,
            tokenizer=tokenizer)
        print("---------------------Training-------------------")
        trainer.train()
        print("---------------------Evaluate-------------------")
        print(trainer.evaluate(datasets[2].map(tokenization, batched=True)))
        trainer.save_model(args.save_model_dir)
        print('---done---')
    elif len(datasets)==2:
        trainer = Trainer(
            model=model,
            args=training_args, 
            train_dataset=datasets[0].map(tokenization, batched=True),
            eval_dataset=datasets[1].map(tokenization, batched=True),
            compute_metrics=compute_metrics,
            tokenizer=tokenizer)
        print("---------------------Training-------------------")
        trainer.train()
        print("---------------------Evaluate-------------------")
        print(trainer.evaluate())
        trainer.save_model(args.save_model_dir)
        print('---done---')
    else:
        print('ERROR: datasets must have at least 2 subsets and at most 3 subsets')

if __name__ == "__main__":
    args = cli()
    datasets = process(args)
    main(args, datasets)
