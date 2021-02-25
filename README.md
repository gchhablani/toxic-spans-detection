# Toxic-Spans-Detection

Repository for our code and experiments on SemEval-2021 Task-5 Toxic Spans Detection. We are still updating this repository and trying to make the code more efficient. We will be posting the updates here whenever possible. We would love to know about any issues found on this repository. Please create an issue for any queries, or you contact us at chhablani.gunjan@gmail.com.

Pre-print: http://arxiv.org/abs/2102.12254

## Updates

- [25 Feb 2021]: Repository is made public.

## Usage

### Setting Up

Install `src` using the following command:

```sh
python setup.py install
```

Install the requirements:

```sh
pip install -r requirements.txt
```

### Baseline Models

**RNNSL**

```sh
python run_baseline_model.py --configs ./configs/rnnsl/default.yaml
```

Running this command will save the predictions for `train`, `trial`, and `test`, and corresponding F<sub>1</sub> scores.

**SpaCy**

```sh
cd baselines
python spacy_tagging.py
```

Running this command will save the predictions for `train`, `trial`, and `test`, and corresponding F<sub>1</sub> scores.

### Training

For BERT-based models, the configurations are present in the `configs` directory. You can choose a config of your liking. An example for `bert_token`:

```sh
python train.py --train ./configs/bert_token/train.yaml --data ./configs/bert_token/dataset.yaml
```

This will save the logs, checkpoints, and the final model at the path specified in `train.yaml`.

### Evaluating

Evaluation is done based on a checkpoint mentioned in `eval.yaml` configurations. Please ensure the correctness of the checkpoint path before continuing.

```sh
python eval.py --eval ./configs/bert_token/eval.yaml
```

Running this command will save the predictions for `train`, `trial`, and `test`, and corresponding F<sub>1</sub> scores.

### Integrated Gradients

You can also apply Integrated Gradients to an example of your choosing. Change the `sample_index` in the configuration corresponding to the example. By default, the `test` dataset is used, but you can also choose to use this on other datasets. Please ensure correctness of the checkpoint path before continuing. An example command to run for `roberta_token` is:

```sh
python run_integrated_gradients.py --config ./configs/integrated_gradients/roberta_token.yaml
```

Running this command will save the `word_importances`, `token_importances` binary files, with format `(words, importances)` and `(tokens, importances)` respectively. A visualization will also be save in a file called `viz.html`.

### Combining Predictions

We combine checkpoint predictions using intersection or union. An example for union of top-3 roberta checkpoints:

```sh
python /src/utils/combine_preds.py ./configs/combine_predictions/union_roberta_token_best_3_ckpts.yaml
```

### Evaluating Prediction Files

You will need ground truths, predictions to be able to run this evaluation. We calculate F<sub>1</sub> scores on the given file using the following command:

```sh
python calculate_f1_scores.py --ground <ground> --preds <preds> --out <out>
```

where `<ground>` is path to the ground truths, `<preds>` is the prediction file, and `<out>` is the path where the scores are to be saved. Running this command will save the F<sub>1</sub> scores to the path.

## Directory Structure

```sh
.
├── baselines
│   ├── __init__.py
│   ├── models.py
│   ├── spacy_tagging.py
├── calculate_f1_scores.py # To calculate F1 scores given preds and grounds
├── configs
│   ├── bert_base_spans
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   └── train.yaml
│   ├── bert_base_token
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   └── train.yaml
│   ├── bert_crf_token
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   └── train.yaml
│   ├── bert_multi_spans
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   └── train.yaml
│   ├── bert_spans
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   └── train.yaml
│   ├── bert_token
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   └── train.yaml
│   ├── bert_token_spans
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   └── train.yaml
│   ├── combine_predictions
│   │   ├── intersection_roberta_token_best_3_ckpts.yaml
│   │   ├── intersection_roberta_token_union_spanbert_spans_best_3_ckpts.yaml
│   │   ├── intersection_spanbert_spans_best_3_ckpts.yaml
│   │   ├── intersection_spanbert_spans_union_roberta_token_best_3_ckpts.yaml
│   │   ├── union_roberta_token_best_3_ckpts.yaml
│   │   ├── union_roberta_token_union_spanbert_spans_best_3_ckpts.yaml
│   │   ├── union_spanbert_spans_best_3_ckpts.yaml
│   │   └── union_spanbert_spans_union_roberta_token_best_3_ckpts.yaml
│   ├── integrated_gradients
│   │   ├── roberta_token.yaml
│   │   └── spanbert_spans.yaml
│   ├── rnnsl
│   │   └── default.yaml
│   ├── rnnsl_tsd_train_trial
│   │   └── default.yaml
│   ├── roberta_base_spans
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   └── train.yaml
│   ├── roberta_base_token
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   └── train.yaml
│   ├── roberta_crf_token
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   └── train.yaml
│   ├── roberta_multi_spans
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   └── train.yaml
│   ├── roberta_spans
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   └── train.yaml
│   ├── roberta_token
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   ├── roberta_token_best_3_ckpts
│   │   │   ├── eval_1500.yaml
│   │   │   ├── eval_4000.yaml
│   │   │   └── eval_5000.yaml
│   │   └── train.yaml
│   ├── roberta_token_spans
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   └── train.yaml
│   ├── roberta_token_tsd_train_trial
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   └── train.yaml
│   ├── spanbert_crf_token
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   └── train.yaml
│   ├── spanbert_multi_spans
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   └── train.yaml
│   ├── spanbert_spans
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   ├── spanbert_spans_best_3_ckpts
│   │   │   ├── eval_2000.yaml
│   │   │   ├── eval_3500.yaml
│   │   │   └── eval_5000.yaml
│   │   └── train.yaml
│   ├── spanbert_spans_tsd_train_trial
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   └── train.yaml
│   ├── spanbert_token
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   └── train.yaml
│   ├── spanbert_token_spans
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   ├── spanbert_token_spans_spans
│   │   │   └── eval.yaml
│   │   ├── spanbert_token_spans_token
│   │   │   └── eval.yaml
│   │   └── train.yaml
│   ├── toxicbert_spans
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   └── train.yaml
│   ├── toxicbert_token
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   └── train.yaml
│   ├── toxicroberta_spans
│   │   ├── dataset.yaml
│   │   ├── eval.yaml
│   │   └── train.yaml
│   └── toxicroberta_token
│       ├── dataset.yaml
│       ├── eval.yaml
│       └── train.yaml
├── data
│   ├── clean_versions (incorrect)
│   │   ├── clean_train.csv
│   │   ├── clean_train_trial.csv
│   │   ├── clean_trial.csv
│   │   └── modified_train.csv
│   ├── tsd_test.csv
│   ├── tsd_test_spans.csv
│   ├── tsd_train.csv
│   ├── tsd_train_trial.csv
│   └── tsd_trial.csv
├── eval.py
├── evaluation
│   ├── fix_spans.py
│   ├── fix_spans_test.py
│   ├── __init__.py
│   ├── metrics.py
│   ├── semeval2021.py
│   └── semeval2021_test.py
├── __init__.py
├── integrated_gradients.py
├── LICENSE
├── notebooks
│   ├── Exploratory Data Analysis & Preprocessing.ipynb
│   └── Span Length, Contiguous Spans stats.ipynb
├── print_predictions.py
├── __pycache__
│   ├── run_baseline_model.cpython-38.pyc
│   └── test_random_stuff.cpython-38.pyc
├── README.md
├── requirements.txt
├── results
├── run_baseline_model.py
├── setup.py
├── src
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── toxic_spans_crf_tokens.py
│   │   ├── toxic_spans_multi_spans.py
│   │   ├── toxic_spans_spans.py
│   │   ├── toxic_spans_tokens.py
│   │   └── toxic_spans_tokens_spans.py
│   ├── models
│   │   ├── auto_models.py
│   │   ├── bert_crf_token.py
│   │   ├── bert_multi_spans.py
│   │   ├── bert_token_spans.py
│   │   ├── __init__.py
│   │   ├── roberta_crf_token.py
│   │   ├── roberta_multi_spans.py
│   │   ├── roberta_token_spans.py
│   │   └── two_layer_nn.py
│   ├── modules
│   │   ├── activations.py
│   │   ├── embeddings.py
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── optimizers.py
│   │   ├── preprocessors.py
│   │   ├── schedulers.py
│   │   └── tokenizers.py
│   ├── trainers
│   │   ├── base_trainer.py
│   │   ├── __init__.py
│   └── utils
│       ├── combine_preds.py
│       ├── configuration.py
│       ├── __init__.py
│       ├── logger.py
│       ├── mapper.py
│       ├── misc.py
│       ├── postprocess_predictions.py
│       └── viz.py
├── ToxicSpans_SemEval21.ipynb
└── train.py
```

## Tasks

### Done

- [x] Add Directory Structure
- [x] Add Usage

### Ongoing

- [ ] Update README
  - [ ] Add Approaches
  - [ ] Add Results and Analysis

### To-Do

- [ ] Fix Linting Issues
- [ ] Update Docs

## Approaches

## Results and Analysis

### Data

| Name            | Value |
| :-------------- | ----: |
| Train Data Size |  7939 |
| Trial Data Size |   690 |
| Test Data Size  |  2000 |

|                                |    Train | Trial |   Test |
| :----------------------------- | -------: | :---: | -----: |
| Spaces Marked as Toxic (sum)   | 13278.00 |  830  | 575.00 |
| Spaces Marked as Toxic (mean)  |     1.67 | 1.20  |   0.29 |
| Spaces Marked as Toxic (std)   |     7.72 | 4.48  |   3.19 |
| Words Cut in Spans (sum)       |      263 |  26   |      8 |
| Words Cut in Spans (mean)      |     0.03 | 0.04  |   0.00 |
| Words Cut in Spans (std)       |     0.20 | 0.23  |   0.06 |
| Spans start/end w space (sum)  |       20 |   1   |      1 |
| Spans start/end w space (mean) |     0.00 | 0.00  |   0.00 |
| Spans start/end w space (std)  |     0.05 | 0.00  |   0.02 |

Based on BERT Tokenizer:

|               |  Train |  Trial |   Test |
| :------------ | -----: | -----: | -----: |
| #Tokens(mean) |   47.5 |   46.1 |  43.12 |
| #Tokens(std)  |  45.46 |  43.82 |  39.88 |
| #Tokens(max)  |    335 |    234 |    291 |
| #Tokens(min)  |      1 |      1 |      2 |
| #Words(mean)  |  35.95 |  35.01 |  32.86 |
| #Words(std)   |  34.97 |  34.42 |  31.01 |
| #Words(max)   |    192 |    182 |    186 |
| #Words(min)   |      1 |      1 |      1 |
| #Chars(mean)  | 204.57 | 199.47 | 186.41 |
| #Chars(std)   | 201.37 | 196.63 | 178.76 |
| #Chars(max)   |   1000 |    998 |   1000 |
| #Chars(min)   |      4 |      5 |      6 |

|                            | Train | Trial |  Test |
| :------------------------- | ----: | ----: | ----: |
| Num Contiguous Spans(Mean) |   1.3 |  1.31 |  0.93 |
| Num Contiguous Spans(Std)  |  0.84 |  0.74 |  0.62 |
| Num Contiguous Spans(Max)  |    25 |     6 |     7 |
| Num Contiguous Spans(Min)  |     0 |     0 |     0 |
| Len Contiguous Spans(Mean) | 13.51 |  11.3 |  7.89 |
| Len Contiguous Spans(Std)  | 38.57 | 20.76 | 17.86 |
| Len Contiguous Spans(Max)  |   994 |   350 |   713 |
| Len Contiguous Spans(Min)  |     1 |     1 |     3 |
| Per Contiguous Spans(Mean) |  0.14 |  0.14 |  0.09 |
| Per Contiguous Spans(Std)  |   0.2 |   0.2 |  0.14 |
| Per Contiguous Spans(Max)  |     1 |     1 |     1 |
| Per Contiguous Spans(Min)  |     0 |     0 |     0 |

## References

## Citation

You can cite our work as:

```sh
@misc{chhablani2021nlrg,
      title={NLRG at SemEval-2021 Task 5: Toxic Spans Detection Leveraging BERT-based Token Classification and Span Prediction Techniques},
      author={Gunjan Chhablani and Yash Bhartia and Abheesht Sharma and Harshit Pandey and Shan Suthaharan},
      year={2021},
      eprint={2102.12254},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

If you use any part of our code in your work, please use the following citation:

```sh
@misc{chhablani2021nlrggithub,
  author = {Gunjan Chhablani and Yash Bhartia and Abheesht Sharma and Harshit Pandey and Shan Suthaharan},
  title = {gchhablani/toxic-spans-detection},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/gchhablani/toxic-spans-detection}},
}
```
