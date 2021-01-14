import pandas as pd
from src.modules.tokenizers import *
from src.modules.embeddings import *
from src.utils.mapper import configmapper

from datasets import Dataset, DatasetDict

class Preprocessor:
	def preprocess(self):
		pass


@configmapper.map("preprocessors", "glove")
class GlovePreprocessor(Preprocessor):
	"""GlovePreprocessor."""

	def __init__(self, config):
		"""
		Args:
			config (src.utils.module.Config): configuration for preprocessor
		"""
		super(GlovePreprocessor, self).__init__()
		self.config = config
		self.tokenizer = configmapper.get_object(
			"tokenizers", self.config.main.preprocessor.tokenizer.name
		)(**self.config.main.preprocessor.tokenizer.init_params.as_dict())
		self.tokenizer_params = (
			self.config.main.preprocessor.tokenizer.init_vector_params.as_dict()
		)

		self.tokenizer.initialize_vectors(**self.tokenizer_params)
		self.embeddings = configmapper.get_object(
			"embeddings", self.config.main.preprocessor.embedding.name
		)(
			self.tokenizer.text_field.vocab.vectors,
			self.tokenizer.text_field.vocab.stoi[self.tokenizer.text_field.pad_token],
		)

	def preprocess(self, model_config, data_config):
		train_dataset = configmapper.get_object("datasets", data_config.main.name)(
			data_config.train, self.tokenizer
		)
		val_dataset = configmapper.get_object("datasets", data_config.main.name)(
			data_config.val, self.tokenizer
		)
		model = configmapper.get_object("models", model_config.name)(
			self.embeddings, **model_config.params.as_dict()
		)

		return model, train_dataset, val_dataset


@configmapper.map("preprocessors","qa_preprocessor")
class QAPreprocessor(Preprocessor):
	"""QAPreprocessor."""
	def __init__(self,config):
		"""
		Args:
			config (src.utils.module.Config): configuration for preprocessor
		"""
		super(QAPreprocessor, self).__init__()
		self.config = config

		self.tokenizer = configmapper.get_object(
			"tokenizers", self.config.main.preprocessor.tokenizer.name
		).from_pretrained(
			**self.config.main.preprocessor.tokenizer.init_params.as_dict()
		)
		self.max_length = None
		self.doc_stride = None
		self.pad_on_right = None

	def preprocess(self, model_config, data_config):

		train_data = configmapper.get_object("datasets", data_config.main.name)(data_config.train).get_dataset()
		val_data = configmapper.get_object("datasets", data_config.main.name)(data_config.val).get_dataset()

		datasets = DatasetDict({'train':train_data,'validation':val_data})

		self.max_length = data_config.main.max_length # 384 # The maximum length of a feature (question and context)
		self.doc_stride = data_config.main.doc_stride # 128 # The authorized overlap between two part of the context when splitting it is needed.
		self.pad_on_right = self.tokenizer.padding_side == "right"

		tokenized_datasets = datasets.map(self.prepare_train_features, batched=True, remove_columns=datasets["train"].column_names)

		model = configmapper.get_object("models", model_config.name).from_pretrained(
            **model_config.params.as_dict()
        )

		return model,tokenized_datasets["train"],tokenized_datasets["validation"]


	def prepare_train_features(self,examples):
		# Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
		# in one example possible giving several features when a context is long, each of those features having a
		# context that overlaps a bit the context of the previous feature.
		print(self)
		print(examples)
		tokenized_examples = self.tokenizer(
			examples["question" if self.pad_on_right else "context"],
			examples["context" if self.pad_on_right else "question"],
			truncation="only_second" if self.pad_on_right else "only_first",
			max_length=self.max_length,
			stride=self.doc_stride,
			return_overflowing_tokens=True,
			return_offsets_mapping=True,
			padding="max_length",
		)

		# Since one example might give us several features if it has a long context, we need a map from a feature to
		# its corresponding example. This key gives us just that.
		sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
		# The offset mappings will give us a map from token to character position in the original context. This will
		# help us compute the start_positions and end_positions.
		offset_mapping = tokenized_examples.pop("offset_mapping")

		# Let's label those examples!
		tokenized_examples["start_positions"] = []
		tokenized_examples["end_positions"] = []
		print("HIIIII")
		for i, offsets in enumerate(offset_mapping):
			# We will label impossible answers with the index of the CLS token.
			input_ids = tokenized_examples["input_ids"][i]
			cls_index = input_ids.index(self.tokenizer.cls_token_id)

			# Grab the sequence corresponding to that example (to know what is the context and what is the question).
			sequence_ids = tokenized_examples.sequence_ids(i)

			# One example can give several spans, this is the index of the example containing this span of text.
			sample_index = sample_mapping[i]
			answers = examples["answers"][sample_index]
			# If no answers are given, set the cls_index as answer.
			if len(answers["answer_start"]) == 0:
				tokenized_examples["start_positions"].append(cls_index)
				tokenized_examples["end_positions"].append(cls_index)
			else:
				# Start/end character index of the answer in the text.
				start_char = answers["answer_start"][0]
				end_char = start_char + len(answers["text"][0])

				# Start token index of the current span in the text.
				token_start_index = 0
				while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
					token_start_index += 1

				# End token index of the current span in the text.
				token_end_index = len(input_ids) - 1
				while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
					token_end_index -= 1

				# Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
				if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
					tokenized_examples["start_positions"].append(cls_index)
					tokenized_examples["end_positions"].append(cls_index)
				else:
					# Otherwise move the token_start_index and token_end_index to the two ends of the answer.
					# Note: we could go after the last offset if the answer is the last word (edge case).
					while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
						token_start_index += 1
					tokenized_examples["start_positions"].append(token_start_index - 1)
					while offsets[token_end_index][1] >= end_char:
						token_end_index -= 1
					tokenized_examples["end_positions"].append(token_end_index + 1)
		print("HII END")
		print(tokenized_examples)
		return tokenized_examples