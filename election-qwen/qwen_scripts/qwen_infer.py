from PIL import Image
import json
import re

def convert_to_conversation(sample):
    image_path = f"flattened/{sample["image"]}"
    instruction = f"How many votes did {sample["party"]} score?"
    try: 
        _ = Image.open(image_path)
    except:
        print(sample["image"])
        return None
    conversation = [
    {"role": "user", "content": [
        {"type": "text", "text": instruction},
        {"type": "image", "image": image_path}
    ]}
    ]
    return conversation
pass


import random
with open("enel645_final_data/final_test.json", "r") as f:
    test_set = json.load(f)


filtered_test_set = []
def filter_test_set(test_set):
    # seen_images = set()
    for example in test_set:
        image = example['image']
        # if image in seen_images:
        #     continue
        # seen_images.add(image)
        try:
            _ = Image.open("flattened/" + image)
            filtered_test_set.append(example)
        except:
            continue


filter_test_set(test_set)



random.Random(42).shuffle(filtered_test_set)


if True:
    from unsloth import FastVisionModel
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit = True, # Set to False for 16bit LoRA
    )
    FastVisionModel.for_inference(model) # Enable for inference!

import csv
from transformers import TextStreamer

with open("write_file.csv", "w") as f:
    fieldnames = ["image", "party", "expected", "predicted"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for example in filtered_test_set:
        image_path = "flattened/" + example["image"]
        party = example["party"]
        messages = convert_to_conversation(example)
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = False)
        inputs = tokenizer(
            Image.open(image_path),
            input_text,
            add_special_tokens = False,
            return_tensors = "pt",
        ).to("cuda")

        # text_streamer = TextStreamer(tokenizer, skip_prompt = True)
        # _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
        #                 use_cache = True, temperature = 1.5, min_p = 0.1)
        generated_ids = model.generate(**inputs, max_new_tokens=128,
                               use_cache=True, temperature=1.5, min_p=0.1)

        # Now, `generated_ids` will contain the token IDs of the generated output.
        # If you want to decode them into a human-readable string:
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        out = re.sub("[^0-9]", "", generated_text)

        writer.writerow({
            "party": party,
            "image": example["image"],
            "expected": example["count"],
            "predicted": out
        })