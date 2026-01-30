from itertools import chain
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GPT2LMHeadModel
from tqdm import tqdm
from torch.utils.data import DataLoader

tokenizer = AutoTokenizer.from_pretrained("gpt2")
#ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")


def group_texts(examples):
    blockSize = 1024
    concatenated = {k: list(chain.from_iterable(examples[k])) for k in examples.keys()}
    
    totalLength = (len(concatenated[list(examples.keys())[0]]) // blockSize) * blockSize
    result = {
        k: [t[i : i + blockSize] for i in range(0, totalLength, blockSize)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result




def Main(num):
    ds = load_dataset("text", data_files=f"gen{num}_data.txt")
    TDs = ds.map(lambda examples: tokenizer(examples["text"], truncation=True, max_length=1024), batched=True, remove_columns=["text"])
    lmDataset = TDs.map(group_texts, batched=True)
    lmDataset.set_format(type='torch', columns=['input_ids', 'labels'])
    trainData = lmDataset['train']
    trainDataloader = DataLoader(
    trainData, 
    batch_size=4, 
    shuffle=True, 
    pin_memory=True     
    )
    device = torch.device("cuda")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = torch.cuda.amp.GradScaler()
    trainDataloader = DataLoader(trainData, batch_size=1, shuffle=True, pin_memory=True)
    accumulation_steps = 8 
    
    model.train()
    for epoch in range(3):
        loop = tqdm(trainDataloader, leave=True)
        optimizer.zero_grad()
        
        for i, batch in enumerate(loop):
            inputIds = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(inputIds, labels=labels)
                loss = outputs.loss / accumulation_steps
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            loop.set_description(f"Epoch [{epoch+1}/3]")
            loop.set_postfix(loss=loss.item() * accumulation_steps)

    torch.save(model.state_dict(), f"model_gen_{num}.pth")