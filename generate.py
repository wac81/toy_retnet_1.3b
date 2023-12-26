import torch
from retnet.modeling_retnet import RetNetModelWithLMHead
from retnet.configuration_retnet import load_config_from_yaml
from transformers import AutoTokenizer,AutoModelForCausalLM
import glob 

cp_lst = glob.glob("./checkpoints/checkpoint-*")
cp_lst.sort()
print(cp_lst[-1] + "/pytorch_model.bin")
device=torch.device("cuda:0")


model = RetNetModelWithLMHead.from_pretrained(cp_lst[-1])
model.to(device)
model.eval()

# Trainer API should save tokenizers too in the same directory
tokenizer = AutoTokenizer.from_pretrained(cp_lst[-1],trust_remote_code=True)


tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/baichuan-7B", trust_remote_code=True, use_fast=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id # unk. we want this to be different from the eos token

while True:
    inp = input("user:")
    inputs = "user:{}\nsystem:".format(inp)
    inputs = tokenizer(inputs, return_tensors='pt',padding=True)
    inputs = inputs.to('cuda:0')
    # parallel forward
    generated = model.generate(**inputs, parallel_compute_prompt=True, max_new_tokens=128,
    top_k=30,  #考虑前k个词概率 30
    top_p=0.8, #越大则候选词越多，多样性越高   0.85
    temperature=0.4 #同样越大概率分布就    0.3
    )
    res = tokenizer.batch_decode(generated)
    print('system:', ''.join(res))

