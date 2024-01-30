import torch
import torch.nn as nn
from transformers import ResNetModel
from transformers import XLMRobertaModel
from roberta import RobertaLayer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TextModel(nn.Module):
    def __init__(self):
        super(TextModel, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.layer = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
        )

    def forward(self, input_ids, attention_mask):
        output = self.roberta(input_ids, attention_mask)
        output = self.layer(output.last_hidden_state)
        return output


class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.resnet = ResNetModel.from_pretrained("microsoft/resnet-152")
        self.image_change = nn.Sequential(
            nn.Linear(2048, 768),
            nn.GELU(),
        )

    def forward(self, image_inputs):
        output = self.resnet(image_inputs)
        output = output.last_hidden_state.view(-1, 2048, 49).permute(0, 2, 1).contiguous()
        image_pooled_output, _ = output.max(1)
        output = self.image_change(image_pooled_output)
        return output.unsqueeze(1)


class Classifier(nn.Module):
    def __init__(self, f):
        super(Classifier, self).__init__()
        self.f = f
        self.linear = nn.Sequential(
            nn.Linear(768*2,3),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(768, 3)
        )


    def forward(self, feature):
        if (self.f =="multi"):
            return self.linear(feature)
        if (self.f == "single"):
            return self.linear2(feature)



class MultimodalModel(nn.Module):

    def __init__(self):
        super(MultimodalModel, self).__init__()
        self.text_embedding = TextModel()
        self.img_embedding = ImageModel()
        self.attention = RobertaLayer(self.text_embedding.roberta.config)
        self.classifier1 = Classifier("single")
        self.classifier2 = Classifier("multi")


    def forward(self, image_input, text_input):
        if (image_input is not None) and (text_input is not None):
            text = self.text_embedding(text_input["input_ids"], text_input["attention_mask"])
            image_features = self.img_embedding(image_input)
            image_text_hidden_state = torch.cat([image_features, text], 1)

            text_attention_mask = text_input.attention_mask
            image_attention_mask = torch.ones((text_attention_mask.size(0), 1)).to(device)
            attention_mask = torch.cat([image_attention_mask, text_attention_mask], 1).unsqueeze(1).unsqueeze(2)

            image_text_attention_state = self.attention(image_text_hidden_state, attention_mask)[0]

            final_output = torch.cat([image_text_attention_state[:, 0, :], image_text_attention_state[:, 1, :]], 1)
            out = self.classifier2(final_output)


            return out

        elif image_input is None:
            text = self.text_embedding(text_input["input_ids"], text_input["attention_mask"])
            attention_mask = text_input.attention_mask.unsqueeze(1).unsqueeze(2)

            attention_state = self.attention(text, attention_mask)[0]
            out = self.classifier1(attention_state[:, 0, :])

            return out

        elif text_input is None:
            image_features = self.img_embedding(image_input)
            attention_mask = torch.ones((image_features.size(0), 1)).to(device)
            attention_state = self.attention(image_features, attention_mask)[0]
            out = self.classifier1(attention_state[:, 0, :])

            return out













