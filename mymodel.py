# from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering

import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_pretrained_bert import BertModel, BertConfig
# from utils import kl_coef
import math


def kl_coef(i):
    # coef for KL annealing
    # reaches 1 at i = 22000
    # https://github.com/kefirski/pytorch_RVAE/blob/master/utils/functional.py
    return (math.tanh((i - 3500) / 1000) + 1) / 2


class DomainDiscriminator(nn.Module):
    def __init__(self, num_classes=6, input_size=768 * 2,
                 hidden_size=768, num_layers=3, dropout=0.1):
        super(DomainDiscriminator, self).__init__()
        self.num_layers = num_layers
        hidden_layers = []
        for i in range(num_layers):
            if i == 0:
                input_dim = input_size
            else:
                input_dim = hidden_size
            hidden_layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(), nn.Dropout(dropout)
            ))
        hidden_layers.append(nn.Linear(hidden_size, num_classes))
        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, x):
        # forward pass
        for i in range(self.num_layers - 1):
            x = self.hidden_layers[i](x)
        logits = self.hidden_layers[-1](x)
        log_prob = F.log_softmax(logits, dim=1)
        return log_prob


class DistilBertForQuestionAnsweringwithClassification(nn.Module):
    def __init__(self, mystr, num_classes=6,
                 num_layers=3, dis_lambda=0.5, concat=False, anneal=False):
        super(DistilBertForQuestionAnsweringwithClassification, self).__init__()

        self.distilbertqa = DistilBertForQuestionAnswering.from_pretrained(
            mystr)
        self.config = self.distilbertqa.config
        self.num_labels = 6  # config.num_labels
        # self.dropout=nn.Dropout(config.seq_classif_dropout)
        # self.classifier=nn.Linear(config.hidden_size, 6)
        self.dis_lambda = dis_lambda
        self.anneal = anneal
        if concat:
            input_size = 2 * self.config.hidden_size
        else:
            input_size = self.config.hidden_size
        self.discriminator = DomainDiscriminator(
            num_classes=num_classes, input_size=input_size, hidden_size=self.config.hidden_size, num_layers=num_layers, dropout=self.config.seq_classif_dropout)
        # Initialize weights and apply final processing
        # self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        dtype="qa",
    ):

        if dtype == "qa":
            qa_loss = self.forward_qa(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                start_positions=start_positions,
                end_positions=end_positions,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            return qa_loss

        elif dtype == "dis":
            assert labels is not None
            dis_loss = self.forward_discriminator(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                start_positions=start_positions,
                end_positions=end_positions,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            return dis_loss

        else:
            return 0

    def forward_qa(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        distilbertqa_output = self.distilbertqa(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            start_positions=start_positions,
            end_positions=end_positions,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # if not return_dict:
        #     output = (distilbertqa_output.start_logits,
        #               distilbertqa_output.end_logits) + distilbertqa_output[1:]
        #     return ((distilbertqa_output.loss,) + output) if total_loss is not None else output

        return distilbertqa_output.loss

    def forward_discriminator(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        with torch.no_grad():
            sequence_output = self.distilbertqa(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                start_positions=start_positions,
                end_positions=end_positions,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # [b, d] : [CLS] representation
            cls_embedding = sequence_output[:, 0]
            hidden = cls_embedding
            # if self.concat:
            #     sep_embedding=self.get_sep_embedding(
            #         input_ids, sequence_output)
            #     hidden=torch.cat(
            #         [cls_embedding, sep_embedding], dim = -1)  # [b, 2*d]
            # else:
            #     hidden=cls_embedding
        log_prob = self.discriminator(hidden.detach())
        criterion = nn.NLLLoss()
        loss = criterion(log_prob, labels)
        return loss
