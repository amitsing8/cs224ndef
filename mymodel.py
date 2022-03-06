# from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering

import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_pretrained_bert import BertModel, BertConfig
# from utils import kl_coef


class DistilBertForQuestionAnsweringwithClassification(nn.Module):
    def __init__(self, mystr):
        super(DistilBertForQuestionAnsweringwithClassification, self).__init__()

        self.distilbertqa = DistilBertForQuestionAnswering.from_pretrained(
            mystr)

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
    ):
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        if not return_dict:
            output = (distilbertqa_output.start_logits,
                      distilbertqa_output.end_logits) + distilbertqa_output[1:]
            return ((distilbertqa_output.loss,) + output) if total_loss is not None else output

        return distilbertqa_output
