import torch

class StoryDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer , combined_texts , max_len):
        self.tokenizer = tokenizer
        self.combined_texts = combined_texts
        self.max_len = max_len

    def __len__(self):
        return len(self.combined_texts)

    def __getitem__(self, idx):
        data = self.tokenizer.batch_encode_plus(
            [self.combined_texts[idx]],
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
            truncation=True
        )
        return {
            "input_ids":data["input_ids"].flatten(),
            "attention_mask":data["attention_mask"].flatten(),
            "labels":data["input_ids"].flatten()
        }