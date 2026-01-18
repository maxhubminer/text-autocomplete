import torch
from torch.nn.utils.rnn import pad_sequence

def evaluate(model, loader, criterion, pad_token_id):
    model.eval()
    val_loss = 0.
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            x_batch = batch["input_ids"]
            y_batch = batch["target_ids"]
            lengths = batch["lengths"]

            logits = model(x_batch, lengths)  # [B, T, V]

            # flatten для loss
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y_batch.view(-1)
            )
            val_loss += loss.item()

            # для accuracy
            preds = torch.argmax(logits, dim=-1)  # [B, T]
            mask = (y_batch != pad_token_id)
            correct += ((preds == y_batch) & mask).sum().item()
            total += mask.sum().item()

    val_loss /= len(loader)
    val_acc = correct / total if total > 0 else 0.0
    return val_loss, val_acc
