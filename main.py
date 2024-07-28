from imports import *
from args import get_args
from text_processing import text_combining, text_cleaning
from dataloader_setup import setup_dataloaders
from model_setup import setup_model_and_optimizer
from training import train_and_evaluate
from generate_story import generate_story


def main():
    arg = get_args()

    DATAPATH = arg.input_text_path

    combined_train_texts = text_combining('train_prompt.txt', 'train_story.txt', DATAPATH)
    combined_valid_texts = text_combining('valid_prompt.txt', 'valid_story.txt', DATAPATH)

    model, tokenizer, optimizer, scheduler, device = setup_model_and_optimizer(arg.model_name, arg.learning_rate, None,
                                                                               arg.num_train_epochs, arg.warmup)

    train_dataloader, valid_dataloader = setup_dataloaders(tokenizer, combined_train_texts, combined_valid_texts,
                                                           arg.train_batch_size, arg.valid_batch_size,
                                                           arg.max_seq_length)

    train_and_evaluate(model, tokenizer, optimizer, scheduler, train_dataloader, valid_dataloader, device,
                       arg.num_train_epochs)

    # Example usage of generating a story
    prompt = combined_valid_texts[300][:combined_valid_texts[300].find('<sep>')]
    target = combined_valid_texts[300][combined_valid_texts[300].find('<sep>') + 5:]
    generate_story(prompt, target, tokenizer, model, device)


if __name__ == "__main__":
    main()
