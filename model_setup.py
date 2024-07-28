import torch
from transformers import T5ForConditionalGeneration , T5Tokenizer
from transformers import AdamW , get_linear_schedule_with_warmup

def setup_model_and_optimizer(model_name , lr , train_dataloader , num_epochs , warmup):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    optimizer = AdamW(model.parameters() , lr = lr)
    total_steps = len(train_dataloader)*num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer ,
                                                num_warmup_steps=int(total_steps*warmup),
                                                num_training_steps=total_steps)
    return model , tokenizer , optimizer , scheduler , device