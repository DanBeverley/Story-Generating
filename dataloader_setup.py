from torch.utils.data import DataLoader
from dataset import StoryDataset

def setup_dataloaders(tokenizer , combined_train_texts , combined_valid_texts , train_batch_size,
                      valid_batch_size , max_seq_length):
    train_dataset = StoryDataset(tokenizer , combined_train_texts , max_seq_length)
    valid_dataset = StoryDataset(tokenizer , combined_valid_texts , max_seq_length)

    train_dataloader = DataLoader(train_dataset , batch_size = train_batch_size , shuffle = True)
    valid_dataloader = DataLoader(valid_dataset , batch_size = valid_batch_size , shuffle = False)

    return train_dataloader , valid_dataloader