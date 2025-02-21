import pandas as pd
import torch
from torch.utils.data import Dataset


class CommentDataset (Dataset):

    def __init__(self, data: pd.DataFrame, tokenizer, max_token_len: int = 200):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        comment = str(item.Review)
        label = torch.FloatTensor(self.data.iloc[index, 1:])
        encoding = self.tokenizer.encode_plus(
                                comment,
                                add_special_tokens=True,
                                max_length=self.max_token_len,
                                return_token_type_ids=False,
                                padding="max_length",
                                truncation=True,
                                return_attention_mask=True,
                                return_tensors='pt'
        )
        return {'input_ids': encoding["input_ids"].flatten(),
                'attention_mask': encoding["attention_mask"].flatten(),
                'label': label}
