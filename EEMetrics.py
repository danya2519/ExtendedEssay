import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset

def calculate_perplexity(model_path):
    device = torch.device("cuda")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    testDs = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    
    encodings = tokenizer("\n\n".join(testDs["text"]), return_tensors="pt")
    
    maxLength = 512 
    stride = 512
    seqLen = encodings.input_ids.size(1)

    nlls = []
    prev_endLoc = 0
    for beginLoc in tqdm(range(0, seqLen, stride)):
        endLoc = min(beginLoc + maxLength, seqLen)
        trgLen = endLoc - prev_endLoc
        inputIds = encodings.input_ids[:, beginLoc:endLoc].to(device)
        targetIds = inputIds.clone()
        targetIds[:, :-trgLen] = -100

        with torch.no_grad():
            outputs = model(inputIds, labels=targetIds)
            negLogLikelihood = outputs.loss

        nlls.append(negLogLikelihood * trgLen)
        prevEndLoc = endLoc
        if endLoc == seqLen:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / endLoc)
    print(f"\nPerplexity for {model_path}: {ppl.item():.2f}")
    return ppl.item()

if __name__ == "__main__":
    previousPpl = 0
    for i in range(0, 9):
        temp = calculate_perplexity(f"model_gen_{i}.pth")
        print(f"diff is {temp - previousPpl}")
        previousPpl = temp