# Toxic-Spans-Detection
Repository for our code and experiments on SemEval-2021 Task-5 Toxic Spans Detection.

## To-Do
- [ ] Run Baselines
- [ ] Literature Survey


## Experiments


### Approaches


## Analysis

### Data

| Name                     | Value |
| :----------------------- | :---- |
| Original Train Data Size | 7939  |
| Original Trial Data Size | 690   |
| Common Examples          | 5     |
| Reduced Train Data Size  | 7934  |


|                                | Reduced Train | Trial |
| :----------------------------- | :------------ | :---- |
| Spaces Marked as Toxic (sum)   | 13278.00      | 830   |
| Spaces Marked as Toxic (mean)  | 1.67          | 1.20  |
| Spaces Marked as Toxic (std)   | 7.72          | 4.48  |
| Words Cut in Spans (sum)       | 263           | 26    |
| Words Cut in Spans (mean)      | 0.03          | 0.04  |
| Words Cut in Spans (std)       | 0.20          | 0.23  |
| Spans start/end w space (sum)  | 20            | 1     |
| Spans start/end w space (mean) | 0.00          | 0.00  |
| Spans start/end w space (std)  | 0.05          | 0.00  |

Fixing using:
```
def find_word_by_character_index(text, idx):
    if(not text[idx].isalnum()):
        backward = idx
        forward = idx
        while(backward>-1 and not text[backward].isalnum()):
            backward-=1
        while(forward<len(text) and text[forward].isalnum()):
            forward+=1

        return text[backward+1:forward], backward+1,forward-1
    else:
        return text[idx], idx, idx

def clean_text(text,contiguous_spans):
    new_contiguous_spans = []

    ## Adding, Removing Cut Words
    for i in contiguous_spans:
        start = i[0] 
        end = i[-1]

        if start==0 and end==len(text)-1:
            new_contiguous_spans.append([start,end])

        elif start==0:
            if text[end].isalnum() and text[end+1].isalnum():

                full_word,full_start,full_end = find_word_by_character_index(text,end)
                cut_word_len = end-full_start+1
                if(cut_word_len*2>=len(full_word)):
                    new_contiguous_spans.append([start,full_end])
                else:
                    new_contiguous_spans.append([start,full_start-1])


        elif i[-1]==len(text)-1:
            if text[start].isalnum() and text[start-1].isalnum():
                full_word, full_start,full_end = find_word_by_character_index(text,start)
                cut_word_len = full_end-start+1
                if(cut_word_len*2>=len(full_word)):
                    new_contiguous_spans.append([full_start,end])
                else:
                    new_contiguous_spans.append([full_end+1,end])
                
                
        else:
            new_start = start
            new_end = end
           
            if text[start].isalnum() and text[start-1].isalnum():
                full_word, full_start,full_end = find_word_by_character_index(text,start)
                cut_word_len = full_end-start+1
                if(cut_word_len*2>=len(full_word)):
                    new_start = full_start
                else:
                    new_start = full_end+1

            if text[end].isalnum() and text[end+1].isalnum():
                full_word, full_start,full_end = find_word_by_character_index(text, end)
                cut_word_len = end-full_start+1
                if(cut_word_len*2>=len(full_word)):
                    new_end = full_end
                else:
                    new_end = full_start-1
            new_contiguous_spans.append([new_start,new_end])
    ## Remove Spaces from span beginning and end

    newest_contiguous_spans = []
    for i in new_contiguous_spans:
        start = i[0]
        end = i[-1]
        while start<=end:
            if(not (text[start].isalnum()) or not (text[end].isalnum())):
                if not (text[start].isalnum()):
                    start+=1
                if not (text[end].isalnum()):
                    end-=1
            else:
                break
        if(start<=end):
            newest_contiguous_spans.append([start,end])
    return newest_contiguous_spans
```

|               | Clean Train | Clean Trial | Reduced Train |  Trial |
| :------------ | ----------: | ----------: | ------------: | -----: |
| #Tokens(mean) |       47.53 |        46.1 |         47.53 |   46.1 |
| #Tokens(std)  |       45.46 |       43.82 |         45.46 |  43.82 |
| #Tokens(max)  |         335 |         234 |           335 |    234 |
| #Tokens(min)  |           1 |           1 |             1 |      1 |
| #Words(mean)  |       35.97 |       35.01 |         35.97 |  35.01 |
| #Words(std)   |       34.97 |       34.42 |         34.97 |  34.42 |
| #Words(max)   |         192 |         182 |           192 |    182 |
| #Words(min)   |           1 |           1 |             1 |      1 |
| #Chars(mean)  |      204.69 |      199.47 |        204.69 | 199.47 |
| #Chars(std)   |      201.37 |      196.63 |        201.37 | 196.63 |
| #Chars(max)   |        1000 |         998 |          1000 |    998 |
| #Chars(min)   |           4 |           5 |             4 |      5 |


|                            | Train | Trial | Modified Train | Clean Train | Clean Trial |
| -------------------------- | ----- | ----- | -------------- | ----------- | ----------- |
| Num Contiguous Spans(Mean) | 1.30  | 1.31  | 1.30           | 1.17        | 1.19        |
| Num Contiguous Spans(Std)  | 0.83  | 0.74  | 0.83           | 0.88        | 0.79        |
| Num Contiguous Spans(Max)  | 25    | 6     | 25             | 25          | 6           |
| Num Contiguous Spans(Min)  | 0     | 0     | 0              | 0           | 0           |
| Len Contiguous Spans(Mean) | 13.51 | 11.30 | 13.51          | 12.06       | 10.28       |
| Len Contiguous Spans(Std)  | 38.57 | 20.76 | 38.58          | 34.51       | 16.36       |
| Len Contiguous Spans(Max)  | 994   | 350   | 994            | 997         | 281         |
| Len Contiguous Spans(Min)  | 1     | 1     | 1              | 1           | 1           |

**Note**: There are five files in the data directory. They are described as follows:
1. tsd_train.csv : Original Training File
2. tsd_trial.csv : Original Trial File
3. modified_train.csv : After removing those samples from tsd_train.csv common in both tsd_train.csv and tsd_trial.csv
4. clean_train.csv : After removing beginning and trailing non-alphanumeric characters from modified_train.csv, and including larger hald words and removing shorter half words (see above code).
5. clean_trial.csv : Same process as clean_train on trial.csv

## Results

## References
### Papers

### Articles
- [ ] https://paperswithcode.com/task/hate-speech-detection
- [ ] https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/#:~:text=the%20input%20layer.-,Start%20%26%20End%20Token%20Classifiers,into%20the%20start%20token%20classifier.
- [ ] https://medium.com/datadriveninvestor/extending-google-bert-as-question-and-answering-model-and-chatbot-e3e7b47b721a#:~:text=Using%20BERT%20for%20Question%20and,labeled%20Question%20and%20answer%20dataset.
- [ ] https://medium.com/sfu-cspmp/detecting-toxic-comment-f309a20a5127
