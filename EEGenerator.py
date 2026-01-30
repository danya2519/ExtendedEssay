import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

def generate_text(model_path, prompt, max_len=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    
    model.load_state_dict(torch.load(model_path))
    model.eval() 

    inputIds = tokenizer.encode(prompt, return_tensors="pt").to(device)

    #print(prompt)
    with torch.no_grad():
        output = model.generate(
            inputIds,
            max_length=max_len,
            do_sample=True,      
            temperature=0.8,     
            top_k=50,            
            top_p=0.95,          
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    print(result)
    return result

if __name__ == "__main__":
    #generate_text("model_gen_0.pth", "The scientific discovery of")
    prompt = input(":")
    for i in range(0, 9):
        print(f"\n--- Gen {i} ---\n")
        generate_text(f"model_gen_{i}.pth", prompt)
        print("\n" + "="*50 + "\n")

def generate_text_main(prompt):
    temp = []
    for num in range(0, 9):
        temp.append(generate_text(f"model_gen_{num}.pth", prompt))
    return temp

