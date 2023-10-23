import torch
from torch import nn
from torchcrf import CRF
from .modeling_bert import BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from torchvision.models import resnet50
import torch.nn.functional as F
from pytorchcv.model_provider import get_model as ptcv_get_model

class SE_ResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super(SE_ResNet50, self).__init__()
        self.seresnet = ptcv_get_model("seresnet50", pretrained=pretrained)

    def forward(self, x):
        prompt_guids = []
        for name, layer in self.seresnet.named_children():
            if name == 'features':
                for name2, module in layer.named_children():
                    if 'pool' in name2:continue
                    x = module(x)
                    if 'stage' in name2:
                        bsz, channel, ft, _ = x.size()
                        kernel = ft // 2
                        prompt_kv = nn.AvgPool2d(kernel_size=(kernel, kernel), stride=kernel)(x)  # (bsz, 2048, 1, 1)
                        prompt_guids.append(prompt_kv)
        return prompt_guids

class SeresnetModel(nn.Module):
    def __init__(self, args):
        super(SeresnetModel, self).__init__()
        self.args = args
        self.seresnet = SE_ResNet50()

    def forward(self, x, aux_imgs=None):

        prompt_guids = self.seresnet(x)
        if aux_imgs is not None:
            aux_prompt_guids = []
            aux_imgs = aux_imgs.permute([1, 0, 2, 3, 4])
            for i in range(len(aux_imgs)):
                aux_prompt_guid = self.seresnet(aux_imgs[i])
                aux_prompt_guids.append(aux_prompt_guid)
            return prompt_guids, aux_prompt_guids
        return prompt_guids, None

class resnetModel(nn.Module):
    def __init__(self, args):
        super(resnetModel, self).__init__()
        self.args = args
        self.resnet = resnet50(pretrained=True)
        self.seresnet = SE_ResNet50()

    def forward(self, x, aux_imgs=None):
        prompt_guids = self.get_resnet_prompt(x)
        if aux_imgs is not None:
            aux_prompt_guids = []
            aux_imgs = aux_imgs.permute([1, 0, 2, 3, 4])
            for i in range(len(aux_imgs)):
                aux_prompt_guid = self.get_resnet_prompt(aux_imgs[i])
                aux_prompt_guids.append(aux_prompt_guid)
            return prompt_guids, aux_prompt_guids
        return prompt_guids, None

    def get_resnet_prompt(self, x):
        prompt_guids = []
        for name, layer in self.resnet.named_children():
            if name == 'fc' or name == 'avgpool':  continue
            x = layer(x)
            if 'layer' in name:
                bsz, channel, ft, _ = x.size()
                kernel = ft // 2
                prompt_kv = nn.AvgPool2d(kernel_size=(kernel, kernel), stride=kernel)(x)
                prompt_guids.append(prompt_kv)
        return prompt_guids

class AttentionREClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionREClassifier, self).__init__()

        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.attention = nn.Linear(input_size, 1)

    def forward(self, x):
        att_scores = self.attention(x)
        att_scores = att_scores.squeeze(-1)
        att_weights = F.softmax(att_scores, dim=1)

        att_x = torch.bmm(x.permute(0, 2, 1), att_weights.unsqueeze(-1))
        att_x = att_x.squeeze(-1)

        output = F.relu(self.fc1(att_x))

        return output

class CREModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(CREModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.args = args

        self.dropout = nn.Dropout(0.5)
        self.classifier = AttentionREClassifier(input_size=768, hidden_size=256, output_size=num_labels)
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer

        if self.args.use_prompt:
            self.image_model = SeresnetModel(args)
            self.encoder_conv = nn.Sequential(
                                    nn.Linear(in_features=3840, out_features=800),
                                    nn.Tanh(),
                                    nn.Linear(in_features=800, out_features=4*2*768)
                                )
            self.gates = nn.ModuleList([nn.Linear(4*768*2, 4) for i in range(12)])
            self.encoder_text = nn.Linear(768, 2*768)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        images=None,
        aux_imgs=None,
        seq = None,
        img = None,
        relation = None,
        seq_input_ids = None,
        seq_token_type_ids = None,
        seq_attention_mask = None
    ):
        seq_output = self.bert(
            input_ids=seq_input_ids,
            token_type_ids=seq_token_type_ids,
            attention_mask=seq_attention_mask,
            output_attentions=True,
            return_dict=True
        )
        seq_last_hidden_state, seq_pooler_output = seq_output.last_hidden_state, seq_output.pooler_output
        bsz = input_ids.size(0)
        if self.args.use_prompt:
            prompt_guids = self.get_visual_prompt(images, aux_imgs, seq_last_hidden_state)
            prompt_guids_length = prompt_guids[0][0].shape[2]
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)
        else:
            prompt_guids = None
            prompt_attention_mask = attention_mask

        output = self.bert(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=prompt_attention_mask,
                    past_key_values=prompt_guids,
                    output_attentions=True,
                    return_dict=True
        )

        last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
        bsz, seq_len, hidden_size = last_hidden_state.shape
        entity_hidden_state = torch.Tensor(bsz, 2, hidden_size)
        for i in range(bsz):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            head_hidden = last_hidden_state[i, head_idx, :].unsqueeze(0)
            tail_hidden = last_hidden_state[i, tail_idx, :].unsqueeze(0)
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=0)
        entity_hidden_state = entity_hidden_state.to(self.args.device)
        logits = self.classifier(entity_hidden_state)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(logits, labels.view(-1)), logits, seq, img, relation
        return logits


class CNERModel(nn.Module):
    def __init__(self, label_list, args):
        super(CNERModel, self).__init__()
        self.args = args
        self.prompt_dim = args.prompt_dim
        self.prompt_len = args.prompt_len
        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert_config = self.bert.config

        if args.use_prompt:
            self.image_model = SeresnetModel(args)
            self.encoder_conv = nn.Sequential(
                            nn.Linear(in_features=3840, out_features=800),
                            nn.Tanh(),
                            nn.Linear(in_features=800, out_features=4*2*768)
                            )
            self.gates = nn.ModuleList([nn.Linear(4*768*2, 4) for i in range(12)])
            self.encoder_text = nn.Linear(768, 2 * 768)
        self.num_labels = len(label_list)
        print(self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None, aux_imgs=None,
                seq_input_ids=None, seq_token_type_ids=None, seq_attention_mask=None):
        if self.args.use_prompt:
            seq_output = self.bert(
                input_ids=seq_input_ids,
                token_type_ids=seq_token_type_ids,
                attention_mask=seq_attention_mask,
                output_attentions=True,
                return_dict=True
            )
            seq_last_hidden_state, seq_pooler_output = seq_output.last_hidden_state, seq_output.pooler_output
            bsz = input_ids.size(0)

            prompt_guids = self.get_visual_prompt(images, aux_imgs, seq_last_hidden_state)
            prompt_guids_length = prompt_guids[0][0].shape[2]
            bsz = attention_mask.size(0)
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)
        else:
            prompt_attention_mask = attention_mask
            prompt_guids = None

        bert_output = self.bert(input_ids=input_ids,
                            attention_mask=prompt_attention_mask,
                            token_type_ids=token_type_ids,
                            past_key_values=prompt_guids,
                            return_dict=True)
        sequence_output = bert_output['last_hidden_state']
        sequence_output = self.dropout(sequence_output)
        emissions = self.fc(sequence_output)
        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, attention_mask.byte(), reduction='mean')

        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )
    def get_visual_prompt(self, images, aux_imgs, seq_last_hidden_state):
        bsz = images.size(0)
        prompt_guids, aux_prompt_guids = self.image_model(images, aux_imgs)  # [bsz, 256, 2, 2], [bsz, 512, 2, 2]....

        prompt_guids = torch.cat(prompt_guids, dim=1).view(bsz, self.args.prompt_len, -1)   # bsz, 4, 3840
        aux_prompt_guids = [torch.cat(aux_prompt_guid, dim=1).view(bsz, self.args.prompt_len, -1) for aux_prompt_guid in aux_prompt_guids]  # 3 x [bsz, 4, 3840]

        prompt_guids = self.encoder_conv(prompt_guids)  # bsz, 4, 4*2*768
        aux_prompt_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in aux_prompt_guids] # 3 x [bsz, 4, 4*2*768]
        split_prompt_guids = prompt_guids.split(768*2, dim=-1)   # 4 x [bsz, 4, 768*2]
        split_aux_prompt_guids = [aux_prompt_guid.split(768*2, dim=-1) for aux_prompt_guid in aux_prompt_guids]   # 3x [4 x [bsz, 4, 768*2]]
        result = []
        for idx in range(12):  # 12
            sum_prompt_guids = torch.stack(split_prompt_guids).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
            prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_prompt_guids)), dim=-1)

            key_val = torch.zeros_like(split_prompt_guids[0]).to(self.args.device)  # bsz, 4, 768*2
            for i in range(4):
                key_val = key_val + torch.einsum('bg,blh->blh', prompt_gate[:, i].view(-1, 1), split_prompt_guids[i])

            aux_key_vals = []   # 3 x [bsz, 4, 768*2]
            for split_aux_prompt_guid in split_aux_prompt_guids:
                sum_aux_prompt_guids = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
                aux_prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_aux_prompt_guids)), dim=-1)
                aux_key_val = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)  # bsz, 4, 768*2
                for i in range(4):
                    aux_key_val = aux_key_val + torch.einsum('bg,blh->blh', aux_prompt_gate[:, i].view(-1, 1), split_aux_prompt_guid[i])
                aux_key_vals.append(aux_key_val)
                key_val = [key_val] + aux_key_vals
                key_val = torch.cat(key_val, dim=1)
            key_val = key_val.split(768, dim=-1)
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1, 64).contiguous()  # bsz, 12, 4, 64
            temp_dict = (key, value)
            result.append(temp_dict)
        return result



