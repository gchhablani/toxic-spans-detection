import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from src.utils.mapper import configmapper


@configmapper.map("models", "bert_token_spans")
class BertModelForTokenAndSpans(BertPreTrainedModel):
    def __init__(self, config, num_token_labels=2, num_qa_labels=2):
        super(BertModelForTokenAndSpans, self).__init__(config)
        self.bert = BertModel(config)
        self.num_token_labels = num_token_labels
        self.num_qa_labels = num_qa_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_token_labels)
        self.qa_outputs = nn.Linear(config.hidden_size, num_qa_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        labels=None,  # Token Wise Labels
        output_attentions=None,
        output_hidden_states=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=None,
        )

        sequence_output = outputs[0]

        qa_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        sequence_output = self.dropout(sequence_output)
        token_logits = self.classifier(sequence_output)

        total_loss = None
        if (
            start_positions is not None
            and end_positions is not None
            and labels is not None
        ):
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = token_logits.view(-1, self.num_token_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                token_loss = loss_fct(active_logits, active_labels)
            else:
                token_loss = loss_fct(
                    token_logits.view(-1, self.num_token_labels), labels.view(-1)
                )

            total_loss = (start_loss + end_loss) / 2 + token_loss

        output = (start_logits, end_logits, token_logits) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output