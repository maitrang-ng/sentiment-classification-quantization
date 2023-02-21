# Buil dataset from csv files.

from datasets import load_dataset, ClassLabel
from constant import train_dataset, test_dataset
from argparse import ArgumentParser

class Dataset:
    def __init__(self):
        pass

    def build_dataset(self, file_paths):
        dataset = load_dataset("csv", data_files=[file_path for file_path in file_paths])
        label = ClassLabel(num_classes = 2 ,names=["neg", "pos"])
        dataset = dataset.cast_column("label", label)
        dataset = dataset.shuffle()
        return dataset['train']

    '''def concat_dataset(self, *datasets):
        dataset = concatenate_datasets([dataset for dataset in datasets])
        result = dataset.shuffle()'''

    def split_dataset(self, dataset, test_size=0.2):
        split_dataset = dataset.train_test_split(test_size=test_size, stratify_by_column='label', shuffle=True, seed=42)
        train_dataset, test_dataset = split_dataset['train'], split_dataset['test']
        return train_dataset, test_dataset
    
def cli():
    parser = ArgumentParser()
    # Arguments users used when running command lines
    parser.add_argument("--dataset", "-ds", nargs="+", type=str, default=[train_dataset, test_dataset],
                        help='Directory of dataset files in order: train, validation, test as a list. Value in list must be string.')
    parser.add_argument("--ratio", "-r", nargs="+", type=float, default=[0.2],
                        help='ratio is used in the case you want concatenat all datasets and then split with your own ratio.')   
    '''parser.add_argument("--shuffle", "-s", type=bool, default=True,
                        help='set shuffle is True or False. Default = True')
    parser.add_argument("--seed", type=int, default=42)'''
    return args

def process(args):
    # Load dataset from path
    ds = Dataset()
    if args.ratio:
        if len(args.ratio)==2:
            dataset = ds.build_dataset(list(args.dataset))
            train_dataset, test_dataset = ds.split_dataset(dataset, args.ratio[0])
            train_dataset, validation_dataset = ds.split_dataset(train_dataset, float(args.ratio[1]/(1-args.ratio[0])))
            print('Train dataset size:', len(train_dataset))
            print('Validation dataset size:', len(validation_dataset))
            print('Test dataset size:', len(test_dataset))
            result = train_dataset, validation_dataset, test_dataset
        elif len(args.ratio)==1:
            dataset = ds.build_dataset(list(args.dataset))
            train_dataset, test_dataset = ds.split_dataset(dataset, args.ratio[0])
            print('Train dataset size:', len(train_dataset))
            print('Test dataset size:', len(test_dataset))
            result = train_dataset, test_dataset
        else:
            print('Split ratio must contain 1 or 2 float values that correspond test split & validation split.')
            result = "error"
    elif not args.ratio:
        if len(args.dataset)==3:
            train_dataset = ds.build_dataset([args.dataset[0]])
            validation_dataset = ds.build_dataset([args.dataset[1]])
            test_dataset = ds.build_dataset([args.dataset[2]])
            print('Train dataset size:', len(train_dataset))
            print('Validation dataset size:', len(validation_dataset))
            print('Test dataset size:', len(test_dataset))
            result = train_dataset, validation_dataset, test_dataset
        elif len(args.dataset)==2:
            train_dataset = ds.build_dataset([args.dataset[0]])
            test_dataset = ds.build_dataset([args.dataset[1]])
            print('Train dataset size:', len(train_dataset))
            print('Test dataset size:', len(test_dataset)) 
            result = train_dataset, test_dataset
        else:
            print('ERROR: Dataset must have at least 2 values for train & test and at most 3 values for train, validation & test.')
            print('If you have 1 or >= 3 dataset files, you can concat & split your dataset by providing test size or validation size or both.')
            result = "error"
    return result

if __name__ == "__main__":
    args = cli()
    process(args)
