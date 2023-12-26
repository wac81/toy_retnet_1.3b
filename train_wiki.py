# python train_wiki.py --output_dir checkpoints --do_train --do_eval --per_device_train_batch_size 1 --per_device_eval_batch_size 1

from dataclasses import dataclass

from transformers import (Trainer, TrainingArguments, AutoTokenizer, HfArgumentParser,
                          DataCollatorForLanguageModeling)
from datasets import load_dataset

from retnet.modeling_retnet import RetNetModelWithLMHead
from retnet.configuration_retnet import load_config_from_yaml
import torch

@dataclass
class MyArgs:
    model_size: str = '1.3b'
    dataset_name: str = 'wiki'
    text_col: str = 'text'
    max_length: int = 1024

def main():
    parser = HfArgumentParser((TrainingArguments, MyArgs))
    
    train_args, args = parser.parse_args_into_dataclasses()
    train_args.save_total_limit = 3
    train_args.gradient_accumulation_steps = 64
    train_args.eval_accumulation_steps = 1
    if args.dataset_name == 'wiki': 
        #sample_by="document" 加载所有文本
        train_dataset = load_dataset('text', data_files={'train': "/nvme/PaLM-rlhf-pytorch/data/zhwiki-latest-pages-articles-multistream.txt"},split='train[5%:95%]') #[5%:95%] 
        eval_dataset = load_dataset('text', 
                                    data_files={'train': "/nvme/PaLM-rlhf-pytorch/data/zhwiki-latest-pages-articles-multistream.txt"},
                                    split='train[:5%]+train[-5%:]')  #train[:5%]+train[-5%:]
    else:
        train_dataset = load_dataset(args.dataset_name, split="train")
        eval_dataset = load_dataset(args.dataset_name, split="validation")

    config = load_config_from_yaml(f"configs/retnet-{args.model_size}.yml")
    model = RetNetModelWithLMHead(config)

    # tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # tokenizer.model_max_length = 16384
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.unk_token = tokenizer.eos_token
    # tokenizer.bos_token = tokenizer.eos_token

    tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/baichuan-7B", trust_remote_code=True, use_fast=True)

    tokenizer.pad_token_id = tokenizer.unk_token_id  # unk. we want this to be different from the eos token


    # tokenizer.pad_token_id = tokenizer.unk_token_id = tokenizer.eos_token_id = 0  # unk. we want this to be different from the eos token
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.unk_token = tokenizer.eos_token
    # tokenizer.bos_token = tokenizer.eos_token


    def tokenize_datset(example):
        # example[args.text_col] = [line for line in example[args.text_col] if len(line) > 0 and not line.isspace()]
        # if len(example[args.text_col]) == 0: return False

        input_ids = tokenizer(example[args.text_col],
                              truncation=True,
                              max_length=args.max_length,
                            #   padding="max_length",
                              return_tensors='pt').input_ids[0]
        return {'input_ids': input_ids} 

    train_dataset = train_dataset.filter(lambda x: len(x["text"]) > 5,num_proc=8)
    eval_dataset = eval_dataset.filter(lambda x: len(x["text"]) > 5,num_proc=8)

    train_dataset = train_dataset.map(tokenize_datset, remove_columns=train_dataset.column_names, num_proc=8)
    eval_dataset = eval_dataset.map(tokenize_datset, remove_columns=eval_dataset.column_names, num_proc=8)


    # model.load_state_dict(torch.load("/nvme/RetNet/checkpoints/checkpoint-70500/pytorch_model.bin"))

    trainer = Trainer(model=model,
                      args=train_args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      tokenizer=tokenizer,
                      data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))

    if train_args.do_train:
        trainer.train()
        trainer.save_model()
    if train_args.do_eval:
        trainer.evaluate()


if __name__ == "__main__":
    main()
