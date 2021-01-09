# Toxic-Spans-Detection
Repository for our code and experiments on SemEval-2021 Task-5 Toxic Spans Detection.

## To-Do
- [ ] Run Baselines
- [ ] Literature Survey


## Experiments


### Approaches


## Analysis

### Data

| Name                       | Value |
|:---------------------------|:------|
| Original Train Data Size   | 7939  |
| Original Trial Data Size   | 690   |
| Common Examples            | 5     |
| Reduced Train Data Size    | 7934  |
| Post-split Train Data Size | 6347  |
| Post-split Dev Data Size   | 1587  |


|                                | Train    | Trial | Dev  |
|:-------------------------------|:---------|:------|:-----|
| Spaces Marked as Toxic (sum)   | 10610.00 | 830   | 2668 |
| Spaces Marked as Toxic (mean)  | 1.67     | 1.20  | 1.68 |
| Spaces Marked as Toxic (std)   | 7.96     | 4.48  | 6.66 |
| Words Cut in Spans (sum)       | 212      | 26    | 51   |
| Words Cut in Spans (mean)      | 0.03     | 0.04  | 0.03 |
| Words Cut in Spans (std)       | 0.20     | 0.23  | 0.21 |
| Spans start/end w space (sum)  | 13       | 1     | 7    |
| Spans start/end w space (mean) | 0.00     | 0.00  | 0.00 |
| Spans start/end w space (std)  | 0.05     | 0.00  | 0.04 |

Removing Samples where Words Cut:

|                            | Train | Trial | Dev  |
|:---------------------------|:------|:------|:-----|
| Data Shape post "Cleaning" | 6162  | 670   | 1646 |


|               | Clean Train | Clean Dev | Clean Trial | Unclean Train | Unclean Dev | Unclean Trial |
|:--------------|------------:|----------:|------------:|--------------:|------------:|--------------:|
| #Tokens(mean) |       47.04 |     47.24 |       45.67 |         47.45 |       47.84 |          46.1 |
| #Tokens(std)  |       45.22 |     44.56 |       43.03 |         45.54 |       45.15 |         43.82 |
| #Tokens(max)  |         299 |       280 |         227 |           299 |         335 |           234 |
| #Tokens(min)  |           1 |         1 |           1 |             1 |           1 |             1 |
| #Words(mean)  |       35.56 |     36.09 |       34.73 |         35.85 |       36.45 |         35.01 |
| #Words(std)   |       34.76 |     34.79 |       33.98 |            35 |       34.85 |         34.42 |
| #Words(max)   |         192 |       191 |         182 |           192 |         191 |           182 |
| #Words(min)   |           1 |         1 |           1 |             1 |           1 |             1 |

## Results

## References
### Papers

### Articles
- [ ] https://paperswithcode.com/task/hate-speech-detection
- [ ] https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/#:~:text=the%20input%20layer.-,Start%20%26%20End%20Token%20Classifiers,into%20the%20start%20token%20classifier.
- [ ] https://medium.com/datadriveninvestor/extending-google-bert-as-question-and-answering-model-and-chatbot-e3e7b47b721a#:~:text=Using%20BERT%20for%20Question%20and,labeled%20Question%20and%20answer%20dataset.
- [ ] https://medium.com/sfu-cspmp/detecting-toxic-comment-f309a20a5127
