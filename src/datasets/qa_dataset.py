import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from src.utils.mapper import configmapper

from evaluation.fix_spans import _contiguous_ranges

@configmapper.map("datasets","qa_dataset")
class QADataset:
    def __init__(self,config):
        self.config = config

        df_old = pd.read_csv(self.config.file_path)
        self.df = self.create_dataframe(df_old)

        self.data = Dataset.from_pandas(self.df)


    def create_dataframe(self,df):
        df_new = pd.DataFrame(columns=['answers', 'context', 'id','question','title'])
        id = 0
        for row_number in range(df.shape[0]):
            row = df.iloc[row_number]
            context = row['text']
            if(row['spans'][0]=='[' and row['spans'][1]==']'):
                continue
            span = row['spans'].strip('][').split(', ')
            span = [int(i) for i in span]
            question = "find offensive spans"
            title = context.split(' ')[0]
            contiguous_spans = _contiguous_ranges(span)

            for lst in contiguous_spans:
                lst = list(lst)
                dict_to_write = {}
                
                dict_to_write['answer_start'] = [lst[0]]
                dict_to_write['text'] = [context[lst[0]:lst[-1]+1]]
                # print(dict_to_write)
                df_new = df_new.append({'answers':dict_to_write, 'context':context,'id':str(id),'question':question,'title':title},ignore_index=True)
                id += 1
        return df_new

    def get_dataset(self):
        return self.data