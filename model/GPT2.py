from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import torch
import tensorflow as tf



if __name__ == '__main__':

    print(tf.__version__)

    tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
    model=GPT2LMHeadModel.from_pretrained('gpt2')


    input_ids=tokenizer.encode('Who was the first president of India', return_tensors='pt')



    greedy_output=model.generate(input_ids,max_length=50, do_sample=False)

    #print(tokenizer.decode(greedy_output[0],skip_special_tokens=True))


