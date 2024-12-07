# -*- coding: utf-8 -*-
#  
### PaliGemma Fine-tuning Inference
#Author: Ziheng Chang
#

# # # Initialization

#import libraries

# Importing Libraries
import os
import sys
import site
from transformers import AutoModelForCausalLM, PaliGemmaProcessor, AutoModelForPreTraining,PaliGemmaForConditionalGeneration
from huggingface_hub import login
import torch
from PIL import Image
import re
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from evaluate import load
cer = load("cer")

# Setup site packages path
python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
site_packages_path = os.path.expanduser(f'~/.local/lib/python{python_version}/site-packages')
site.addsitedir(site_packages_path)

# Login to Hugging Face Hub
token = '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' #Your own hugging face token
login(token=token)

#  
# # # Load data. 
#

answer, state, code = np.genfromtxt('test_data.txt', delimiter = ', ', unpack = True, dtype=str)
print(answer[0], state[0], code[0])

#Function to load and format the data as needed
directory = '/home/ziheng.chang/election_images/images/'
formatted_data = []
PU_Name = None

for i in range(len(answer)):
    image = (Image.open(directory+state[i]+'/'+code[i]+'.jpg.jpg')).resize((448,448))
    formatted_entry = {'multiple_choice_answer': answer[i],'question': f'How many votes did APC score?','image': image,
                       'state': state[i], 'code': code[i]}
    formatted_data.append(formatted_entry)

print(len(formatted_data), 'images loaded.')
print(formatted_data[0])

            
#Load our trained model
model_id = "palicoqiqi/paligemma_ocr_final"
#model_id = 'google/paligemma-3b-mix-448'
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
processor = PaliGemmaProcessor.from_pretrained('google/paligemma-3b-mix-448') #('google/paligemma-3b-mix-448')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loading and Processing the Image



def make_predictions(examples):
    predicted = []
    actual = []
    count = 0
    tot_count = len(examples)
    print('incorrect examples:')
    for example in examples:
        texts = "<image> <bos>" + example["question"]
        labels= example['multiple_choice_answer']
        images = example["image"].convert("RGB")
        
        # Preprocessing Inputs
        inputs = processor(text=texts, images=images, padding="longest", do_convert_rgb=True, return_tensors="pt").to(device)
        inputs = inputs.to(dtype=model.dtype)
        
        # Generating and Decoding Output
        with torch.no_grad():
            output = model.generate(**inputs, max_length=2000)

        out = re.sub("[^0-9]", "", processor.decode(output[0], skip_special_tokens=True))
        
        predicted.append(out)
        actual.append(labels)

        if out != labels and count < 200:
            print('-------------------------------------')
            print('predicts',out)
            print('actually', labels)
            print(example["code"])
            plt.imshow(images)
            plt.show()
            
        if count % 100 == 0:
            Accuracy = metrics.accuracy_score(actual, predicted)
            print(count/tot_count*100, '% done. So far accuracy is', Accuracy*100, '%. CER is',
                  cer.compute(predictions=predicted, references=actual)*100, '%.')
        count += 1

        #if count > 200:
        #    break
        
        #print(processor.decode(output[0], skip_special_tokens=True),', and actually is:',labels)
    print('Completed. Accuracy is', metrics.accuracy_score(actual, predicted)*100, '%.',
          '. CER is',cer.compute(predictions=predicted, references=actual)*100, '%.')
    return actual, predicted

actual, predicted = make_predictions(formatted_data)

with open('predictions.txt', 'w') as f:
    for i in range(len(actual)):
        f.write(f"{actual[i]}, ")
        f.write(f"{predicted[i]}\n")