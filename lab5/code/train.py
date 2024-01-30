import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import numpy as np
from model import MultimodalModel
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(120)
torch.cuda.manual_seed(120)
torch.cuda.manual_seed_all(120)
np.random.seed(120)
torch.backends.cudnn.deterministic = True

def model_train(train_dataloader, valid_dataloader, args):

    model = MultimodalModel()
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(lr=args.lr, params=optimizer_grouped_parameters)
    criterion = CrossEntropyLoss()
    num_training_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=15,
                                                num_training_steps=num_training_steps)
    best_rate = 0
    print('start training')

    for epoch in range(args.epochs):
        total_loss, correct, total = 0, 0, 0
        target_list, pred_list = [], []
        model.train()
        for (guid, tag, image, text) in train_dataloader:
            tag, image, text = tag.to(device), image.to(device), text.to(device)
            if args.multi_or_single=="text":
                output = model(image_input=None, text_input=text)
            elif args.multi_or_single=="image":
                output = model(image_input=image, text_input=None)
            else:
                output = model(image_input=image, text_input=text)

            loss = criterion(output, tag)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            total_loss += loss.item() * len(guid)
            pred = torch.argmax(output, dim=1)
            total += len(guid)
            correct += (pred == tag).sum()

            target_list.extend(tag.cpu().tolist())
            pred_list.extend(pred.cpu().tolist())

        total_loss /= total
        accuracy = accuracy_score(target_list, pred_list)
        precision = precision_score(target_list, pred_list, average='macro')
        recall = recall_score(target_list, pred_list, average='macro')
        f1 = f1_score(target_list, pred_list, average='macro')
        print('Train epoch: {0}, Loss: {1}, Accuracy: {2},Precision: {3},Recall: {4},F1: {5}\n'.format(epoch + 1, total_loss, accuracy, precision,
                                                                                           recall, f1))

        total_loss, correct, total = 0, 0, 0
        target_list, pred_list = [], []
        model.eval()

        for guid, tag, image, text in valid_dataloader:
            tag = tag.to(device)
            image = image.to(device)
            text = text.to(device)

            if args.multi_or_single=="text":
                output = model(image_input=None, text_input=text)
            elif args.multi_or_single=="image":
                output = model(image_input=image, text_input=None)
            else:
                output = model(image_input=image, text_input=text)

            loss = criterion(output, tag)

            total_loss += loss.item() * len(guid)
            pred = torch.argmax(output, dim=1)
            total += len(guid)
            correct += (pred == tag).sum()

            target_list.extend(tag.cpu().tolist())
            pred_list.extend(pred.cpu().tolist())

        total_loss /= total

        rate = correct / total * 100
        accuracy = accuracy_score(target_list, pred_list)
        precision = precision_score(target_list, pred_list, average='macro')
        recall = recall_score(target_list, pred_list, average='macro')
        f1 = f1_score(target_list, pred_list, average='macro')
        print('val epoch: {0}, Loss: {1}, Accuracy: {2},Precision: {3},Recall: {4},F1: {5}\n'.format(epoch + 1, total_loss, accuracy, precision,
                                                                                           recall, f1))

        if rate > best_rate:
            best_rate = rate
            print('best rate:{:.2f}%'.format(rate))
            torch.save(model.state_dict(), 'best_model.pth')


def model_test(test_dataloader, args):

    model = MultimodalModel()
    model.load_state_dict(torch.load('best_model.pth'))
    model.to(device)
    print('start testing')
    guid_list = []
    pred_list = []
    model.eval()

    for guid, tag, image, text in test_dataloader:
        image = image.to(device)
        text = text.to(device)

        if args.multi_or_single == "text":
            output = model(image_input=None, text_input=text)
        elif args.multi_or_single == "image":
            output = model(image_input=image, text_input=None)
        else:
            output = model(image_input=image, text_input=text)

        pred = torch.argmax(output, dim=1)
        guid_list.extend(guid)
        pred_list.extend(pred.cpu().tolist())

    pred_mapped = {
        0: 'negative',
        1: 'neutral',
        2: 'positive',
    }
    with open('../data/predicts.txt', 'w', encoding='utf-8') as f:
        f.write('guid,tag\n')
        for guid, pred in zip(guid_list, pred_list):
            f.write(f'{guid},{pred_mapped[pred]}\n')
        f.close()


