from PIL import Image
import json
import random
import evaluate
# import wandb 

from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
import torchvision



import os
os.environ["WANDB_PROJECT"]="qwen2_election_counter"

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

    "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
    "unsloth/llava-1.5-7b-hf-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-2B-Instruct-unsloth-bnb-4bit",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 12,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 12,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 42,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"
                      "gate_proj", "up_proj", "down_proj",],
    task_type="CAUSAL_LM"
)

def convert_to_conversation(sample):
    image_path = f"flattened/{sample["image"]}"
    try: 
        _ = Image.open(image_path)
    except:
        print(sample["image"])
        return None
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : f"how many votes did {sample['party']} score"},
            {"type" : "image", "image" : image_path }]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["count"]} ]
        },
    ]
    return { "messages" : conversation }
pass

with open("enel645_final_data/final_train.json", "r") as f:
    train_set = json.load(f)

with open("enel645_final_data/final_test.json", "r") as f:
    test_set = json.load(f)

with open("enel645_final_data/final_val.json", "r") as f:
    val_set = json.load(f)


model_test_set = [t for sample in test_set if (t:=convert_to_conversation(sample))]
train_set = [t for sample in train_set if (t:=convert_to_conversation(sample))]
val_set = [t for sample in val_set if (t:=convert_to_conversation(sample))]

random.Random(42).shuffle(train_set)
train_set = train_set
random.Random(42).shuffle(val_set)
val_set = val_set

# converted_train_set_sampled = converted_train_set[:6000]

# converted_val_set_sampled = converted_val_set[:2000]

print("TRAIN SET", len(train_set))
print("VAL SET", len(val_set))

print("TEST SET", len(test_set))


FastVisionModel.for_inference(model) # Enable for inference!
# f"flattened/{sample["image"]}"
print(test_set[2]["image"])
image = Image.open("flattened/" + test_set[2]["image"])
instruction = f"How many votes did the {test_set[2]["party"]} score?"

print(test_set[2])
print(instruction)

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)


from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig



from jiwer import cer


# generated_ids = model.generate(**inputs, max_new_tokens=128)


# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_texts = tokenizer.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )


cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    # Get predictions and references
    predictions = pred.predictions
    labels = pred.label_ids

    print(predictions)
    print(labels)

    # Convert predictions to text if using tokenized data
    decoded_preds = [tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=False) for p in predictions]
    decoded_labels = [tokenizer.decode(l,  skip_special_tokens=True, clean_up_tokenization_spaces=False) for l in labels]

    # Compute CER
    result = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"cer": result}


FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = train_set,
    eval_dataset = val_set, 
    # compute_metrics=compute_metrics,

    args = SFTConfig(
        evaluation_strategy="epoch",
        # eval_steps=400,
        eval_accumulation_steps = 4, 
        per_device_train_batch_size = 2,
        per_device_eval_batch_size = 2,
        gradient_accumulation_steps = 2,
        warmup_steps = 5,
        # max_steps = 2000,
        do_eval = True, 
        num_train_epochs = 2, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        # metric_for_best_model="cer",         # Specify CER as the key metric
        # greater_is_better=False,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "wandb",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
    ),
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()


#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")




FastVisionModel.for_inference(model) # Enable for inference!

# image = dataset[2]["image"]
# instruction = "Write the LaTeX representation for this image."

print(test_set[2]["image"])
image = Image.open("flattened/" + test_set[2]["image"])
instruction = f"How many votes did the {test_set[2]["party"]} score?"

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)

model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")