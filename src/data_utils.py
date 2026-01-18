# импортируем библиотеки, которые пригодятся для задачи
import re
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# функция для "чистки" текстов
import re

def clean_string(text):
    # нижний регистр
    text = text.lower()

    # 1. ссылки → <url>
    text = re.sub(r'http\S+|www\.\S+', ' <url> ', text)

    # 2. ники пользователей → <user>
    text = re.sub(r'@\w+', ' <user> ', text)

    # 3. эмоции в звёздочках → <emotion>
    text = re.sub(r'\*[^*]+\*', ' <emotion> ', text)

    # 4. удаляем всё, кроме латиницы, цифр, пробелов и < >
    text = re.sub(r'[^a-z0-9\s<>]', '', text)

    # 5. нормализация пробелов
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def save_dataset(texts, filename):
    
    csv_path = os.path.join("data", filename)

    df = pd.DataFrame({"text": texts})
    df.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"Saved {len(texts)} texts to {csv_path}")
    return csv_path

# # кастомная функция collate_fn для формирования батчей
# def collate_fn(batch):
#     texts = [torch.tensor(item['input_ids']) for item in batch]
#     target = [torch.tensor(item['target_ids']) for item in batch]
#     lengths = torch.tensor([len(seq) for seq in texts])
#     padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)


#     return {
#         'input_ids': padded_texts,
#         'target_ids': target,
#         'lengths': lengths, 
#     }

def collate_fn(batch, pad_token_id=0):
    texts = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
    targets = [torch.tensor(item['target_ids'], dtype=torch.long) for item in batch]

    lengths = torch.tensor(
        [max(1, len(seq)) for seq in texts],
        dtype=torch.long
    )

    padded_texts = pad_sequence(
        texts,
        batch_first=True,
        padding_value=pad_token_id
    )

    padded_targets = pad_sequence(
        targets,
        batch_first=True,
        padding_value=pad_token_id
    )

    return {
        'input_ids': padded_texts,     # Tensor [B, T]
        'target_ids': padded_targets,      # Tensor [B, T]
        'lengths': lengths             # Tensor [B]
    }


class BertDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=7):
        self.samples = []
        for line in texts:
            token_ids = tokenizer.encode(line, add_special_tokens=False, max_length=512, truncation=True)
            if len(token_ids) < seq_len:
                continue
            for i in range(1, len(token_ids) - 1):
                context = token_ids[max(0, i - seq_len//2): i] + [tokenizer.mask_token_id] + token_ids[i+1: i+1+seq_len//2]
                if len(context) < seq_len:
                    continue
                target = token_ids[i]
                self.samples.append((context, target))
           
    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)
