import torch

def generate_story(prompt, target, tokenizer, model, device):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
        outputs = model.generate(inputs, max_length=512)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f'Prompt: {prompt}')
    print(f'Target: {target}')
    print(f'Generated: {generated}')
