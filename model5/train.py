# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:41:19 2021

@author: prakh
"""

""

# Code reference: https://github.com/ramsrigouthamg/Paraphrase-any-question-with-T5-Text-To-Text-Transfer-Transformer-
# Referenced T5 : https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html
#                 https://github.com/google-research/text-to-text-transfer-transformer#released-model-checkpoints
"""
Requirements:
torch==1.4.0
transformers==2.9.0
pytorch_lightning==0.7.5
"""
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
"""
Steps to follow:
    1. This time I trained my own model using the reference : https://colab.research.google.com/drive/176NSaYjc2eeI-78oLH_F9-YV3po3qQQO?usp=sharing#scrollTo=brPOSAkjNP5t
    2. Useful link : https://github.com/ramsrigouthamg/Paraphrase-any-question-with-T5-Text-To-Text-Transfer-Transformer-
    3. https://github.com/huggingface/transformers
    4. https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html
    5  I splitted the train file into train and eval(validation file) for model train
    6. Downloaded the weights of model
    7. Use the weights as similiar to model1
"""


def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

#In particular here, your model file should be named pytorch_model.bin

# Model 2
# model = T5ForConditionalGeneration.from_pretrained('prakhar_t5_base/') # Model 2 with training set splitted in 80:20 ratio for train/val and run for 2 epochs
# Model 3
model = T5ForConditionalGeneration.from_pretrained('prakhar_t5_model5/') # Model 5 with training set splitted in 95:5 ratio for train/val and run for 3 epochs

tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_paraphraser')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
model = model.to(device)


def paraphrase(sentence):
    text =  "paraphrase: " + sentence + " </s>"
    max_len = 100
    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)


    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    beam_outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            do_sample=True,
            max_length=100,
            top_k=120,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=10
        )

    print ("\nOriginal Question :: ",sentence)
    print ("Paraphrased Questions :: ")
    final_outputs =[]
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)

    for i, final_output in enumerate(final_outputs):
        print("{}: {}".format(i, final_output))
        print("\n")
        return final_output
    
    
    
import pandas as pd
file =('eval.csv') #file for evaulating/testing
df = pd.read_csv(file,header=None) 
df[1] = df.apply (lambda row: paraphrase(row[0]), axis=1)
# df.to_csv('file_model2.csv') 
df.to_csv('file_model5.csv', index=False) 