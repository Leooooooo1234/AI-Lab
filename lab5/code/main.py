import pandas as pd
import argparse
from data_processor import get_dataloader, process_dataset
from train import model_train, model_test


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--multi_or_single', default="image")
    parser.add_argument('--test_or_train', default="train")
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--weight_decay', default=5e-3,type=float)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()
    train_data = pd.read_csv("../data/train.txt")
    test_data = pd.read_csv("../data/test_without_label.txt")

    train_list, test_list = process_dataset(train_data, test_data)
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(train_list, test_list, args.batch_size)

    if args.test_or_train == "train":
        model_train(train_dataloader, valid_dataloader, args)
    if args.test_or_train == "test":
        model_test(test_dataloader, args)

