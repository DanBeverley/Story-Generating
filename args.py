import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=88888)
    parser.add_argument("--model_name", default="google-t5/t5-base", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--valid_batch_size", default=32, type=int)
    parser.add_argument("--num_train_epochs", default=20, type=int)
    parser.add_argument("--warmup", default=.1, type=float)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--input_text_path", default="/kaggle/input/story-text", type=str)
    return parser.parse_args()