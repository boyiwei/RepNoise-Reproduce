

import json
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


DEVICE = 0 if torch.cuda.is_available() else -1

INDEX_COL_NAME = 'sample_id'
INPUT_COL_NAME = 'content'
TARGET_COL_NAME = 'score'

classification_map = {
    False: 0, # is harmful
    True: 1, # is safe
}

PROMPT_TEMPLATE = """
Question: {question}
Answer: {answer}
"""

ANSWER_ONLY_TEMPLATE  = """
Answer: {answer}
"""

TEST_BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 32

if __name__ == '__main__':
    model_name = "deberta-v3-xsmall"
    tokenizer = AutoTokenizer.from_pretrained(f"microsoft/{model_name}")

    with open('./data/random_answers.json') as f:
        random_answers = json.load(f)

    training_ds = load_dataset("/home/bw1822/representation-noising/BeaverTails", split='330k_train')
    test_ds = load_dataset("/home/bw1822/representation-noising/BeaverTails", split='30k_test')
    
    def preprocess_function(examples):
        questions = examples["prompt"]
        answers = examples["response"]
        examples['labels'] = [
            classification_map[is_safe] for is_safe in examples['is_safe']
        ]
        content = []
        # randomly use empy 5% of the time and use random tokens 5% of the time
        use_empty = np.random.rand(len(questions)) < 0.05
        use_random_tokens = np.random.rand(len(questions)) < 0.05
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if use_empty[i]:
                examples['labels'][i] = 1
                answer = ""
                content.append(
                    PROMPT_TEMPLATE.format(answer=answer, question=question)
                )
            elif use_random_tokens[i]:
                examples['labels'][i] = 1
                rand_selection = np.random.randint(0, len(random_answers))
                answer = random_answers[rand_selection]
                content.append(
                    PROMPT_TEMPLATE.format(answer=answer, question=question)
                )
            else:
                content.append(
                    PROMPT_TEMPLATE.format(answer=answer, question=question)
                )
        examples['content'] = content
        
        return tokenizer(examples["content"], padding=True, truncation=True, max_length=1024)

    tokenized_train = training_ds.map(preprocess_function, batched=True)
    tokenized_train = tokenized_train.remove_columns(
        ['prompt', 'response', 'category', 'is_safe']
    )
    tokenized_train = tokenized_train.class_encode_column("labels")

    tokenized_test = test_ds.map(preprocess_function, batched=True)
    tokenized_test = tokenized_test.remove_columns(
        ['prompt', 'response', 'category', 'is_safe']
    )
    tokenized_test = tokenized_test.class_encode_column("labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        f"microsoft/{model_name}",
        num_labels=2,
    )

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            # micro and macro, f1, precision, recall
            "macro_f1": f1_score(labels, predictions, average='macro'),
            "macro_precision": precision_score(labels, predictions, average='macro'),
            "macro_recall": recall_score(labels, predictions, average='macro'),
            "micro_f1": f1_score(labels, predictions, average='micro'),
            "micro_precision": precision_score(labels, predictions, average='micro'),
            "micro_recall": recall_score(labels, predictions, average='micro'),
        }
    training_args = TrainingArguments(
        output_dir=f"{model_name}-beavertails-harmful-qa-classifier",
        learning_rate=6e-6,
        per_device_train_batch_size=TEST_BATCH_SIZE,
        per_device_eval_batch_size=TRAIN_BATCH_SIZE,
        num_train_epochs=4,
        weight_decay=0.01,
        warmup_steps=1000,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True # switch off if not using GPU
    )

    print("Training model")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print(trainer.evaluate())
    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(f"{model_name}-beavertails-harmful-qa-classifier")
    trainer.push_to_hub(
        f"{model_name}-beavertails-harmful-qa-classifier"
    )
