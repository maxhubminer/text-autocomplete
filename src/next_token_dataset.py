from datasets import load_dataset
from torch.utils.data import Dataset

# загружаем датасет
def load_sentiment140():
    dataset = load_dataset("sentiment140", split="train", trust_remote_code=True)
    texts = [line for line in dataset["text"]]
    return texts

class TextCompletionDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512, mode='train'):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        assert mode in ['train', 'inference']
        self.mode = mode

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False
        ).input_ids

        if len(tokens) < 2:
            tokens = [
                self.tokenizer.pad_token_id,
                self.tokenizer.pad_token_id
            ]

        if self.mode == 'train':
            # Language modeling: input → target = next token
            if len(tokens) < 2:
                input_ids = tokens
                target_ids = tokens
            else:
                input_ids = tokens[:-1]
                target_ids = tokens[1:]

        elif self.mode == 'inference':
            # Autocompletion: 3/4 input → 1/4 target
            split_idx = int(len(tokens) * 0.75)
            input_ids  = tokens[:split_idx]
            target_ids = tokens[split_idx:]

        return {
            "input_ids": input_ids,
            "target_ids": target_ids
        }
