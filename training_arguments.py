import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import json
import torch


@dataclass
class TrainingArguments:
    output_dir = "model"
    tokenizer_path = "tokenizer.model"
    corpus_path = "corpus.txt"
    evaluate_during_training = True
    train_batch_size = 1
    eval_batch_size = 1
    block_size = 256
    n_eval_batch = 10
    learning_rate = 1e-4
    max_grad_norm = 1.0
    num_train_epochs = 1
    warmup_steps = 0
    logging_steps = 50
    save_steps = 100
    load = False
    device = "cuda"
