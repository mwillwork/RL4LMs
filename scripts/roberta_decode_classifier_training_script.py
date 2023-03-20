##########
### Does not work!!!
##########

import random
import numpy as np
import torch
from parlai.tasks.decode.agents import DecodeTeacher
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import evaluate
from transformers import TrainingArguments, Trainer
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import os

os.environ["WANDB_DISABLED"] = "true"

class CustomTrainer2(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        #print(f"Custom trainer computing loss: {loss}, outputs: {outputs}, inputs: {inputs}")
        #if not self.model.training:
        #   import pdb;pdb.set_trace()
        return (loss, outputs) if return_outputs else loss


def process_data_into_hf_format(t, output_json):
  print(f"Processsing {t.num_episodes()} episodes.")
  epoch_done = False
  context = ""
  episode_idx = 0
  ex_idx = 0
  while episode_idx < t.num_episodes():
    # if episode_idx % 1000 == 0 and ex_idx == 0:
    #    print(f"Processing episode: {episode_idx}, example: {ex_idx}")
    ex = t.get(episode_idx, ex_idx)
    ex_idx += 1
    if ex["episode_done"]:
        episode_idx += 1
        ex_idx = 0
        if episode_idx == t.num_episodes() - 1:
            # last episode so break
            break
    if "turn_id" not in ex:
        print(f"Warning skipping malformed example: {ex}")
        continue

    if ex["turn_id"] == 0:
        context = ""
    else:
        context = context + ("\n" if len(context) > 0 else "") + ex["text"]

    l = ex['labels'][0]
    if l != 'none':
        if l == "non_contradiction":
          processed_label = 1
        elif l == "contradiction":
          processed_label = 0
        else:
          raise Exception(f"invalid label: {l}")
        output_json["label"].append(processed_label)
        output_json["text"].append(context)
  
  print(output_json["text"][:5]) 
  print(f"Found {len(output_json['label'])} total examples")
  print(f"Found {np.sum(output_json['label'])} non contradictions in the dataset.")


def tokenize_dataset(raw_dataset):
    # Assumes that it is a JSON object that has two keys 
    # "label" and "text" with arrays inside each
    tokenized_dataset = []
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    for idx, ex_text in enumerate(raw_dataset["text"]):
        tokenized_json = tokenizer(ex_text, padding="max_length", truncation=True)
        tokenized_json["labels"] = [raw_dataset["label"][idx]]
        tokenized_dataset.append(tokenized_json)
    return tokenized_dataset


my_metric_acc = evaluate.load("accuracy")
def compute_metrics_v16(eval_pred_obj):
    logits, labels = eval_pred_obj
    predictions = np.argmax(logits, axis=-1)
    labels = eval_pred_obj.label_ids

    # TODO: worried about the label_ids actually being the labels
    # print(f"compute_metrics - predictions: {predictions}, labels: {labels}, sum preds: {np.sum(predictions)}, sum labels: {np.sum(labels)}, len: {len(labels)}")
    eval_object = {"predictions": predictions, "references": labels}
    return my_metric_acc.compute(predictions=predictions, references=eval_pred_obj.label_ids)


def training_loop(model):
    lr_array = [1e-2, 1e-3, 1e-4, 5e-4, 1e-5]
    # CHECKPOINTS_PATH = '/content/drive/Othercomputers/My MacBook Pro/Google Drive/cs234/project/models'
    for lr in lr_array:
        print(f"Starting lr: {lr}")
        training_args = TrainingArguments(
            output_dir=f"test_trainer_{lr}",
            evaluation_strategy="steps",
            eval_steps=100,  # TODO: change!
            save_strategy="steps",
            save_steps=100, # TODO: change!
            per_device_train_batch_size=16,
            learning_rate=lr,
            num_epochs=5,
            # resume_from_checkpoint=f"test_trainer_0.001/checkpoint-1500/",
            bf16=True
        )

    # training_args.set_lr_scheduler(name="cosine", warmup_ratio=0.05, num_epochs=1)
    # training_args.set_lr_scheduler(warmup_ratio=0.05, num_epochs=5)

        custom_trainer2 = CustomTrainer2(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            compute_metrics=compute_metrics_v16
        )
        custom_trainer2.train()





def setup():
    print("IN SETUP")
    train_opt = {"datapath": "roberta_classifier", "datatype": "train"}
    train_teacher = DecodeTeacher(opt=train_opt)
    train_dataset = {"text": [], "label": []}
    process_data_into_hf_format(train_teacher, train_dataset)


    eval_opt = {"datapath": "roberta_classifier", "datatype": "valid"}
    eval_teacher = DecodeTeacher(opt=eval_opt)
    eval_dataset = {"text": [], "label": []}
    process_data_into_hf_format(eval_teacher, eval_dataset)    

    tokenized_train_dataset = tokenize_dataset(train_dataset)
    tokenized_eval_dataset = tokenize_dataset(eval_dataset)

    random.seed(42)
    random.shuffle(tokenized_train_dataset)
    random.shuffle(tokenized_eval_dataset)

    model = AutoModelForSequenceClassification.from_pretrained("roberta-large", num_labels=2)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train()

    LR = 5e-6
    # CHECKPOINTS_PATH = '/content/drive/Othercomputers/My MacBook Pro/Google Drive/cs234/project/models'

    training_args = TrainingArguments(
        output_dir=f"test_trainer_5e-06_continue", 
        evaluation_strategy="steps",
        eval_steps=200,  # TODO: change!
        save_strategy="steps",
        save_steps=200, # TODO: change!
        per_device_train_batch_size=16,
        learning_rate=LR,
        bf16=True,
        num_train_epochs=5,
        warmup_ratio=0.1
    )

    # training_args.set_lr_scheduler(name="cosine", warmup_ratio=0.05, num_epochs=1)
    # training_args.set_lr_scheduler(warmup_ratio=0.05, num_epochs=5)

    custom_trainer2 = CustomTrainer2(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics_v16
    )
    custom_trainer2.train(resume_from_checkpoint=True)


def final_eval(model_path, is_test=True):
    import evaluate

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    datatype = "test" if is_test else "valid"
    print(f"Setting up and tokenizing {datatype} eval split")

    eval_opt = {"datapath": "roberta_classifier", "datatype": datatype}
    eval_teacher = DecodeTeacher(opt=eval_opt)
    eval_dataset = {"text": [], "label": []}
    process_data_into_hf_format(eval_teacher, eval_dataset)
   
    tokenized_eval_dataset = tokenize_dataset(eval_dataset)

    random.seed(42)
    random.shuffle(tokenized_eval_dataset)

    acc_metric = evaluate.load("accuracy")
    p_metric = evaluate.load("precision")
    r_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    trained_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    trained_model.to(device)
    trained_model.eval()
   

    ## If you want to try a specific string
    # test_str = "And have you taken anything for it ?   Not at all. Well, for whatever reason. \n Well, I guess my mother"
    # tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    # test_tokenized = tokenizer([test_str])
    # test_tokenized = {k: torch.tensor(v).to(torch.device("cuda")) for k,v in test_tokenized.items()}
    # classifier_outputs = trained_model(**test_tokenized)
    # torch.nn.functional.softmax(classifier_outputs.logits)


    for idx, batch in enumerate(tokenized_eval_dataset):
        batch = {k: torch.tensor(v).to(device) for k, v in dict(batch).items()}
        with torch.no_grad():
            sz = batch['input_ids'].shape
            if len(sz) == 1:
                batch['input_ids'] =  batch['input_ids'].unsqueeze(0)
                batch['attention_mask'] = batch['attention_mask'].unsqueeze(0)

            outputs = trained_model(**batch)
            if idx % 100 == 0:
                print(f"idx: {idx}, output: {outputs}")

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        acc_metric.add_batch(predictions=predictions, references=batch["labels"])
        p_metric.add_batch(predictions=predictions, references=batch["labels"])
        r_metric.add_batch(predictions=predictions, references=batch["labels"])
        f1_metric.add_batch(predictions=predictions, references=batch["labels"])

    acc = acc_metric.compute()
    p = p_metric.compute()
    r = r_metric.compute()
    f1 = f1_metric.compute()
    print(f"acc: {acc}, p: {p}, r: {r}, f1: {f1}")


if __name__ == "__main__":
    print("IN MAIN")
    # setup()
    final_eval(
        '/home/ubuntu/roberta_classifier/test_trainer_5e-06_continue/checkpoint-400/',
        is_test=False)
