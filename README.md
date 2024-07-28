# Story Generation with Transformers

This project uses a transformer model to generate stories based on given prompts. The model is trained using text data to learn the structure and content required to produce coherent stories.

## Overview
The project is structured as follows:

- **imports.py**: Contains all necessary imports.
- **args.py**: Argument parsing for command-line execution.
- **text_processing.py**: Functions for text processing, combining, and cleaning.
- **dataset.py**: Dataset-related operations.
- **dataloader_setup.py**: Setup for data loaders.
- **model_setup.py**: Model and optimizer setup.
- **training.py**: Training and evaluation functions.
- **generate_story.py**: Functions to generate stories based on prompts.
- **main.py**: Main script to run the project.

## Model Used
The project uses the `transformers` library by Hugging Face. Specifically, it employs a pre-trained T5 model by Google

## Requirements
The project dependencies are listed in the `requirements.txt` file. Install them using:
```bash
pip install -r requirements.txt
