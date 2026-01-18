
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

# Основной цикл обучения
def train(model, train_loader, val_loader, criterion, optimizer, evaluate, pad_token_id, n_epochs = 3):

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.
        for batch in tqdm(train_loader):
            x_batch = batch["input_ids"]
            y_batch = batch["target_ids"]
            lengths = batch["lengths"]            
            optimizer.zero_grad()

            logits = model(x_batch, lengths)  # [B, T, V]

            loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    y_batch.view(-1)
                )

            #loss = criterion(model(x_batch, lengths), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()


        train_loss /= len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion, pad_token_id)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_acc:.2%}")