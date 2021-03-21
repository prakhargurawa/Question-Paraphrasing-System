# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 19:13:50 2021

@author: prakh
"""
# Import relevant libraries
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer

"""
Requirements:
torch==1.4.0
transformers==2.9.0
pytorch_lightning==0.7.5

Model 1:
    1. Created an environment "t5_env" with following spec: torch==1.4.0,transformers==2.9.0,pytorch_lightning==0.7.5
    2. https://towardsdatascience.com/paraphrase-any-question-with-t5-text-to-text-transfer-transformer-pretrained-model-and-cbb9e35f1555
    3. conda activate t5_env
    4. python train.py
"""

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# https://huggingface.co/welcome <-- Your model will then be accessible through its identifier: username/model_name
"""
Anyone can load it from code:
    tokenizer = AutoTokenizer.from_pretrained("username/model_name")
    model = AutoModel.from_pretrained("username/model_name")
"""

model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')
tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_paraphraser')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
model = model.to(device)


def paraphrase(sentence):
    text =  "paraphrase: " + sentence + " </s>"
    max_len = 256
    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 10
    beam_outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            do_sample=True,
            max_length=256,
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
        # we will return the top result but for future this code can be modified to provide top n results
        return final_output
    
# Here we are not training any new model just using a pretrained model which is trained on Quora dataset and it gives
# already good results as the quora dataset is very huge  can be found in folder quora_dataset      
import pandas as pd
file =('eval.csv') #file for evaulating/testing
df = pd.read_csv(file,header=None) 
df[1] = df.apply (lambda row: paraphrase(row[0]), axis=1)
df.to_csv('file_output.csv') 