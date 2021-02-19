import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from src.utils.mapper import configmapper


@configmapper.map("models", "roberta_multi_spans")
class RobertaForMultiSpans(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaForMultiSpans, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.num_labels = config.num_labels

        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
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
        output_attentions=None,
        output_hidden_states=None,
    ):
        outputs = self.roberta(
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

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)  # batch_size
        # print(start_logits.shape, end_logits.shape, start_positions.shape, end_positions.shape)

        total_loss = None
        if (
            start_positions is not None and end_positions is not None
        ):  # [batch_size/seq_length]
            # # If we are on multi-GPU, split add a dimension
            # if len(start_positions.size()) > 1:
            #     start_positions = start_positions.squeeze(-1)
            # if len(end_positions.size()) > 1:
            #     end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            # ignored_index = start_logits.size(1)
            # start_positions.clamp_(0, ignored_index)
            # end_positions.clamp_(0, ignored_index)

            # start_positions = start_logits.view()

            loss_fct = BCEWithLogitsLoss()

            start_loss = loss = loss_fct(
                start_logits,
                start_positions.float(),
            )
            end_loss = loss = loss_fct(
                end_logits,
                end_positions.float(),
            )
            total_loss = (start_loss + end_loss) / 2

        output = (start_logits, end_logits) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output