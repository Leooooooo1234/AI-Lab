from transformers import AutoFeatureExtractor, AutoTokenizer
from PIL import Image
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-152")


def process_file(data):

    data_list = []
    label_dict = {"positive": 2, "neutral": 1, "negative": 0}

    for guid, label in data.values:
        data_folder = "../data/source"
        image_file = f"{data_folder}/{int(guid)}.jpg"
        text_file = f"{data_folder}/{int(guid)}.txt"
        img = Image.open(image_file)
        image = feature_extractor(img, return_tensors="pt")

        img.close()

        with open(text_file, 'rb') as f:
            comment = f.readline().strip()
            for encoding in ['utf-8', 'GBK', 'gb18030']:
                try:
                    comment = comment.decode(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
        f.close()
        label = label_dict.get(label, 0)
        data_list.append({"guid":guid, 'text': comment, 'tag': label, 'image': image})


    return data_list


def process_dataset(train_data, test_data):
    train_list = process_file(train_data)
    test_list = process_file(test_data)

    return train_list, test_list


def collate_fn(data_list):
    tag = [data['tag'] for data in data_list]
    guid = [data['guid'] for data in data_list]

    image = torch.stack([data['image']["pixel_values"] for data in data_list])
    image = torch.squeeze(image, dim=1)

    text = [data['text'] for data in data_list]
    text = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=30)

    if tag[0] is None:
        batch_tag = None
    else:
        batch_tag = torch.LongTensor(tag)

    return guid, batch_tag, image, text


def get_dataloader(train_data_list, test_data_list, batch_size):
    train_dataset, valid_dataset = train_test_split(train_data_list, test_size=0.2, random_state=66)

    train_dataloader = DataLoader(dataset=train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data_list, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader


