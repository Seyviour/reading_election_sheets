# -*- coding: utf-8 -*-
#  
### PaliGemma Fine-tuning
#Author: Ziheng Chang
#

# # # Initialization

#import libraries

import os
from PIL import Image
#from pathlib import Path
#import string
import random
from transformers import Trainer
#import numpy as np
import pandas as pd
from transformers import PaliGemmaForConditionalGeneration
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from transformers import PaliGemmaProcessor
import torch
import re

os.environ["HF_TOKEN"] = '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' #hugging face token
device = "cuda"

#  
# # # Load data. 
#

df = pd.read_csv('filtered.csv', usecols = ['State','PU-Code','PU-Name','APC','LP','PDP'])

#Load and format the data as needed
directory = '/home/ziheng.chang/election_images/images/'
formatted_data = []
PU_Name = None

print('Started data loading...')

for i in range(df.shape[0]):
    #if i > 500: ################### for testing
    #    break
    if i % 2000 == 0:
        print(i/586.02, '% done')
    if df.at[i, 'PU-Name'] == PU_Name:
        #print(i, 'passed because of duplication')
        pass
    elif df.at[i, 'APC'] == 0:
        #print(i, 'passed because of empty results')
        pass
    else:
        PU_Name = df.at[i, 'PU-Name']
        PU_Code = df.at[i, 'PU-Code']
        State = df.at[i, 'State']
        try:
            image = (Image.open(directory+State+'/'+PU_Code+'.jpg.jpg')).resize((448,448))
            party = 'APC' #to be changed
            APC = df.at[i, party]
            answer = str(APC)
            
            formatted_entry = {'multiple_choice_answer': answer,'question': f'How many votes did APC score?','image': image,
                               'state': State, 'code': PU_Code}
            formatted_data.append(formatted_entry)
        except:
            #print(PU_Code+'.jpg missing')
            pass
            
print(len(formatted_data), 'images loaded.')
random.shuffle(formatted_data)
print(formatted_data[0])

#Load the train and evaluation data
train_data = formatted_data[:int(len(formatted_data)*0.8)]
random.shuffle(train_data)
val_data = formatted_data[int(len(formatted_data)*0.8):int(len(formatted_data)*0.9)]
random.shuffle(val_data)
test_data = formatted_data[int(len(formatted_data)*0.9):]
random.shuffle(test_data)

#save the test data for later
with open('test_data.txt', 'w') as f:
    for d in test_data:
        answer = d['multiple_choice_answer']
        state = d['state']
        code = d['code']
        f.write(f"{answer}, ")
        f.write(f"{state}, ")
        f.write(f"{code}\n")

print(len(train_data),len(val_data),len(test_data))

# # # Preprocess data 
# Load the processor to preprocess the dataset.
model_id = 'google/paligemma-3b-mix-448'
processor = PaliGemmaProcessor.from_pretrained(model_id)

#  
# Function to reprocess the dataset with a prompt template, pass it with batches of images to processor to convert to tokens.
def collate_fn(examples):
  texts = ["<image> <bos>" + example["question"] for example in examples]
  labels= [example['multiple_choice_answer'] + '<eos>' for example in examples]
  images = [example["image"].convert("RGB") for example in examples]
  tokens = processor(text=texts, images=images, suffix=labels,
                    return_tensors="pt", padding="longest")
  tokens = tokens.to(torch.bfloat16).to(device)
  return tokens


# # # Load the model
# LoRA & QLoRA fine-tuning to reduce cost of training
bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id,
                                                          quantization_config=bnb_config,
                                                          device_map={"":0})
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# # # Setup training, then train
#Initialize the `TrainingArguments`.

# Reasonable parameters are chosen after experimentation.
# 12 epochs gives good results without running for too long.
# 8 batch size gives good speed withou OOM
# learning rate of 4e-5 gives best accuracy

from transformers import TrainingArguments
args=TrainingArguments(
            num_train_epochs=2, 
            remove_unused_columns=False,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=1, 
            gradient_accumulation_steps=2,
            warmup_steps=2,
            learning_rate=0.00004,
            weight_decay=1e-6,
            adam_beta2=0.999,
            optim="adamw_hf",
            save_strategy="epoch",
            eval_strategy="epoch",
            push_to_hub=True,
            save_total_limit=1,
            output_dir="paligemma_ocr_final",
            bf16=True,
            report_to=["tensorboard"],
            dataloader_pin_memory=False,
            load_best_model_at_end=True,
            eval_do_concat_batches = False
        )

 
# We can now start training.
trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collate_fn,
        args=args
        )
        
print('Starting training...')
trainer.train()

#Push the model to huggingface hub for future use.
#Also save it locally just in case
trainer.push_to_hub('palicoqiqi/paligemma_ocr_final')
trainer.save_model("model/paligemma_ocr_final")

#######################
##### Inference

# Importing Libraries
import sys
import site
#from peft import PeftModel, PeftConfig
#from transformers import AutoModelForCausalLM, PaliGemmaProcessor, AutoModelForPreTraining,PaliGemmaForConditionalGeneration
from huggingface_hub import login
#import requests
from sklearn import metrics
#import matplotlib.pyplot as plt

# Setup site packages path
python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
site_packages_path = os.path.expanduser(f'~/.local/lib/python{python_version}/site-packages')
site.addsitedir(site_packages_path)

# Login to Hugging Face Hub
token = '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' #Your own hugging face token
login(token=token)

#Load our trained model
model_id = "palicoqiqi/paligemma_ocr_final"
#model_id = 'google/paligemma-3b-mix-448'
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
processor = PaliGemmaProcessor.from_pretrained('google/paligemma-3b-mix-448') #('google/paligemma-3b-mix-448')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def make_predictions(examples):
    predicted = []
    actual = []
    count = 0
    tot_count = len(examples)
    for example in examples:
        texts = "<image> <bos>" + example["question"]
        labels= example['multiple_choice_answer']
        images = example["image"].convert("RGB")
        
        # Preprocessing Inputs
        inputs = processor(text=texts, images=images, padding="longest", do_convert_rgb=True, return_tensors="pt").to(device)
        inputs = inputs.to(dtype=model.dtype)
        
        # Generating and Decoding Output
        with torch.no_grad():
            output = model.generate(**inputs, max_length=4500)

        out = re.sub("[^0-9]", "", processor.decode(output[0], skip_special_tokens=True))
          
        if out == labels:
            #print('correct')
            predicted.append(1)
            actual.append(1)
        else:
            #print('incorrect')
            predicted.append(0)
            actual.append(1)
            #print(example["code"])
            #plt.imshow(images)
            #plt.show()

        if count % 200 == 0:
            Accuracy = metrics.accuracy_score(actual, predicted)
            print(count/tot_count*100, '% done. Accuracy so far is', Accuracy*100, '%.')
            print('Last predicted: ', out, ', truth is:', labels)
        count += 1

    print('Completed. Accuracy is', metrics.accuracy_score(actual, predicted)*100, '%.')
    return actual, predicted

actual, predicted = make_predictions(test_data)