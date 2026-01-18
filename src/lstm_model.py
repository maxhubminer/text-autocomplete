import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, vocab_size)


    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        out = self.fc(output)
        return out

    def generate(self, input_ids, max_new_tokens, tokenizer):
        self.eval()
        device = next(self.parameters()).device

        if len(input_ids) == 0:
            input_ids = torch.tensor([tokenizer.pad_token_id], dtype=torch.long)

        input_ids = input_ids.unsqueeze(0).to(device)  # [1, T]
        generated = input_ids.tolist()[0]

        hidden = None
        with torch.no_grad():
            for _ in range(max_new_tokens):
                emb = self.embedding(input_ids[:, -1:])
                out, hidden = self.rnn(emb, hidden)
                logits = self.fc(out)

                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                generated.append(next_token.item())

                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        return tokenizer.decode(generated, skip_special_tokens=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())