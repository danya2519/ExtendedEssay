import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def create_synthetic_dataset(model_path, output_filename, num_samples=5000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    rawDs = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    seeds = [line for line in rawDs["text"] if len(line.split()) > 10][:num_samples]

    
    with open(output_filename, "w", encoding="utf-8") as f:
        for i, seed in enumerate(tqdm(seeds)):
            prompt = " ".join(seed.split()[:5])
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_length=200,      
                    do_sample=True, 
                    temperature=0.8,      
                    top_k=50,             
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )

            generatedText = tokenizer.decode(output[0], skip_special_tokens=True)
            
            f.write(generatedText.replace("\n", " ") + "\n")

    print(f"done. {output_filename}")
 

def Main(num):
    create_synthetic_dataset(f"model_gen_{num}.pth", f"gen{num+1}_data.txt")