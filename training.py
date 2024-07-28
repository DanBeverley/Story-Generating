import torch
import numpy as np
import math
from tqdm.auto import tqdm

def train_and_evaluate(model, tokenizer, optimizer, scheduler, train_dataloader, valid_dataloader, device, num_train_epochs):
    model.to(device)

    for epoch in range(num_train_epochs):
        print(f'Start epoch {epoch+1} of {num_train_epochs}')
        train_loss = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        model.train()
        model.zero_grad()
        for _, inputs in enumerate(epoch_iterator):
            d1, d2, d3 = inputs['input_ids'], inputs['attention_mask'], inputs['labels']
            d1, d2, d3 = d1.to(device), d2.to(device), d3.to(device)
            output = model(input_ids=d1, attention_mask=d2, labels=d3)
            batch_loss = output.loss
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            train_loss += batch_loss.item()
            epoch_iterator.set_description(f'(batch_loss = {batch_loss.item()})')
            del batch_loss
        print(f'Average train loss per example = {train_loss/len(train_dataloader)} in epoch {epoch+1}')
        print(f'Starting evaluate after epoch {epoch+1}')
        eval_loss = []
        model.eval()
        for inputs in tqdm(valid_dataloader, desc="eval"):
            d1, d2, d3 = inputs['input_ids'], inputs['attention_mask'], inputs['labels']
            d1, d2, d3 = d1.to(device), d2.to(device), d3.to(device)
            with torch.no_grad():
                output = model(input_ids=d1, attention_mask=d2, labels=d3)
                batch_loss = output.loss
            eval_loss.append(batch_loss.cpu().item())
            del batch_loss
        eval_loss = np.mean(eval_loss)
        perplexity = math.exp(eval_loss)
        print(f'Average valid loss per example = {eval_loss} in epoch {epoch+1}')
        print(f'Perplexity for valid dataset in epoch {epoch+1} is {perplexity}')
