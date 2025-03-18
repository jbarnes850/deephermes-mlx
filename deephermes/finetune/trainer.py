#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training utilities for DeepHermes fine-tuning with LoRA.

This module provides functions for training MLX models with LoRA.
"""
import os
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Model parameters
    model_name_or_path: str = "mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16"
    lora_rank: int = 8
    lora_alpha: float = 16
    lora_dropout: float = 0.0
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 1
    epochs: int = 3
    max_seq_len: int = 512
    
    # Optimizer parameters
    optimizer: str = "adam"
    beta1: float = 0.9
    beta2: float = 0.999
    
    # Logging and saving
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    output_dir: str = "./output"
    
    # Dataset parameters
    train_file: Optional[str] = None
    eval_file: Optional[str] = None
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None


def format_instruction(example: Dict[str, str]) -> str:
    """
    Format an example from the Alpaca dataset into an instruction.
    
    Args:
        example: Dictionary containing 'instruction', 'input', and 'output' keys
        
    Returns:
        Formatted instruction string
    """
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    return prompt


def load_alpaca_dataset(file_path: str, max_samples: Optional[int] = None) -> List[str]:
    """
    Load the Alpaca dataset from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        max_samples: Maximum number of samples to load
        
    Returns:
        List of formatted instruction strings
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    
    if max_samples is not None:
        data = data[:max_samples]
    
    return [format_instruction(example) for example in data]


def prepare_batch(
    tokenizer: Any,
    texts: List[str],
    max_seq_len: int = 512,
) -> Dict[str, mx.array]:
    """
    Prepare a batch of texts for training.
    
    Args:
        tokenizer: Tokenizer to use
        texts: List of texts to tokenize
        max_seq_len: Maximum sequence length
        
    Returns:
        Dictionary containing input_ids and labels
    """
    # Tokenize texts
    tokens = []
    for text in texts:
        token_ids = tokenizer.encode(text)
        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len]
        tokens.append(token_ids)
    
    # Pad sequences
    max_len = max(len(t) for t in tokens)
    padded_tokens = []
    
    # Get pad token ID (handle different tokenizer interfaces)
    pad_id = getattr(tokenizer, "pad_id", None)
    if pad_id is None:
        # Try alternative attribute names
        pad_id = getattr(tokenizer, "pad_token_id", 0)
    
    for t in tokens:
        padded = t + [pad_id] * (max_len - len(t))
        padded_tokens.append(padded)
    
    # Convert to MLX arrays
    input_ids = mx.array(padded_tokens)
    
    # For causal language modeling, labels are the same as input_ids
    labels = input_ids.copy()
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (input_ids != pad_id).astype(mx.int32)
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def iterate_batches(
    texts: List[str],
    tokenizer: Any,
    batch_size: int,
    max_seq_len: int,
    shuffle: bool = True,
) -> Tuple[Dict[str, mx.array], mx.array]:
    """
    Iterate over batches of texts.
    
    Args:
        texts: List of texts to iterate over
        tokenizer: Tokenizer to use
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        shuffle: Whether to shuffle the data
        
    Yields:
        Tuple of (batch, lengths)
    """
    # Shuffle data if requested
    if shuffle:
        indices = np.random.permutation(len(texts))
        texts = [texts[i] for i in indices]
    
    # Iterate over batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch = prepare_batch(tokenizer, batch_texts, max_seq_len)
        
        # Calculate sequence lengths (excluding padding)
        lengths = (batch["input_ids"] != tokenizer.pad_id).sum(axis=1)
        
        yield batch, lengths


def compute_loss(
    model: nn.Module,
    batch: Dict[str, mx.array],
) -> Tuple[mx.array, Dict[str, Any]]:
    """
    Compute the loss for a batch.
    
    Args:
        model: Model to use
        batch: Batch of data
        
    Returns:
        Tuple of (loss, metrics)
    """
    # Forward pass
    logits = model(batch["input_ids"])
    
    # Calculate loss
    labels = batch["labels"]
    
    # Create a mask to ignore padding tokens
    mask = (labels != -100) & (labels != model.vocab_size - 1)
    
    # Calculate cross-entropy loss
    loss = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    )
    
    # Apply mask and calculate mean
    loss = (loss * mask.reshape(-1)).sum() / mask.sum()
    
    # Calculate perplexity
    perplexity = mx.exp(loss)
    
    # Count number of tokens (excluding padding)
    ntoks = mask.sum().item()
    
    # Return loss and metrics
    return loss, {
        "loss": loss.item(),
        "perplexity": perplexity.item(),
        "ntoks": ntoks,
    }


def train_model(
    model: nn.Module,
    tokenizer: Any,
    train_dataset: List[str],
    eval_dataset: Optional[List[str]] = None,
    config: TrainingConfig = TrainingConfig(),
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train a model with LoRA.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer to use
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        config: Training configuration
        
    Returns:
        Tuple of (trained model, metrics)
    """
    logger.info(f"Starting training with {len(train_dataset)} examples")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize optimizer
    if config.optimizer.lower() == "adam":
        # Check if MLX Adam supports weight_decay
        try:
            optimizer = optim.Adam(
                learning_rate=config.learning_rate,
                betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay,
            )
        except TypeError:
            # If weight_decay is not supported, use without it
            optimizer = optim.Adam(
                learning_rate=config.learning_rate,
                betas=(config.beta1, config.beta2),
            )
            if config.weight_decay > 0:
                logger.warning("Weight decay not supported by MLX Adam optimizer, ignoring")
    elif config.optimizer.lower() == "sgd":
        try:
            optimizer = optim.SGD(
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        except TypeError:
            optimizer = optim.SGD(
                learning_rate=config.learning_rate,
            )
            if config.weight_decay > 0:
                logger.warning("Weight decay not supported by MLX SGD optimizer, ignoring")
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    
    # Create optimizer state
    try:
        parameter_count = sum(p.size for p in model.parameters().values())
        logger.info(f"Model has {parameter_count:,} parameters")
    except Exception as e:
        logger.warning(f"Could not calculate parameter counts: {e}")
        logger.warning("Continuing with training...")
    
    # Define loss function
    def loss_fn(model, batch):
        loss, _ = compute_loss(model, batch)
        return loss
    
    # Create training step function
    @mx.compile
    def train_step(model, batch, opt_state):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, batch)
        opt_state = optimizer.update(opt_state, grads)
        return loss, opt_state
    
    # Create evaluation function
    @mx.compile
    def eval_step(model, batch):
        loss, metrics = compute_loss(model, batch)
        return loss, metrics
    
    # Initialize optimizer state
    opt_state = optimizer.init(model)
    
    # Initialize metrics
    metrics = {
        "train_loss": [],
        "eval_loss": [],
        "train_perplexity": [],
        "eval_perplexity": [],
    }
    
    # Perform initial evaluation
    if eval_dataset:
        logger.info("Performing initial evaluation...")
        eval_losses = []
        for batch, _ in iterate_batches(
            eval_dataset,
            tokenizer,
            config.batch_size,
            config.max_seq_len,
            shuffle=False,
        ):
            loss, _ = eval_step(model, batch)
            eval_losses.append(loss.item())
        
        initial_eval_loss = sum(eval_losses) / len(eval_losses)
        logger.info(f"Initial eval loss: {initial_eval_loss:.4f}")
        metrics["eval_loss"].append(initial_eval_loss)
        metrics["eval_perplexity"].append(mx.exp(mx.array(initial_eval_loss)).item())
    
    # Training loop
    global_step = 0
    total_tokens = 0
    start_time = time.time()
    
    for epoch in range(config.epochs):
        logger.info(f"Epoch {epoch+1}/{config.epochs}")
        
        # Iterate over batches
        for batch, lengths in iterate_batches(
            train_dataset,
            tokenizer,
            config.batch_size,
            config.max_seq_len,
            shuffle=True,
        ):
            # Train step
            loss, opt_state = train_step(model, batch, opt_state)
            
            # Update model parameters
            model = optimizer.update_params(model, opt_state)
            
            # Update metrics
            metrics["train_loss"].append(loss.item())
            metrics["train_perplexity"].append(mx.exp(loss).item())
            
            # Update counters
            global_step += 1
            batch_tokens = lengths.sum().item()
            total_tokens += batch_tokens
            
            # Calculate throughput
            elapsed = time.time() - start_time
            throughput = total_tokens / elapsed if elapsed > 0 else 0
            
            # Log progress
            if global_step % config.logging_steps == 0:
                logger.info(
                    f"Step {global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {config.learning_rate:.8f} | "
                    f"It/sec: {global_step / elapsed:.2f} | "
                    f"Tokens/sec: {throughput:.2f}"
                )
            
            # Evaluate
            if eval_dataset and global_step % config.eval_steps == 0:
                logger.info(f"Performing evaluation at step {global_step}...")
                eval_losses = []
                for eval_batch, _ in iterate_batches(
                    eval_dataset,
                    tokenizer,
                    config.batch_size,
                    config.max_seq_len,
                    shuffle=False,
                ):
                    eval_loss, _ = eval_step(model, eval_batch)
                    eval_losses.append(eval_loss.item())
                
                avg_eval_loss = sum(eval_losses) / len(eval_losses)
                logger.info(f"Eval loss: {avg_eval_loss:.4f}")
                metrics["eval_loss"].append(avg_eval_loss)
                metrics["eval_perplexity"].append(mx.exp(mx.array(avg_eval_loss)).item())
            
            # Save checkpoint
            if global_step % config.save_steps == 0:
                checkpoint_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_path, exist_ok=True)
                
                # Save model weights
                from deephermes.finetune.lora import save_lora_weights
                save_lora_weights(model, os.path.join(checkpoint_path, "lora_weights.npz"))
                
                # Save config
                with open(os.path.join(checkpoint_path, "config.json"), "w") as f:
                    json.dump(vars(config), f, indent=2)
                
                logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(config.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    
    # Save model weights
    from deephermes.finetune.lora import save_lora_weights
    save_lora_weights(model, os.path.join(final_path, "lora_weights.npz"))
    
    # Save config
    with open(os.path.join(final_path, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2)
    
    logger.info(f"Saved final model to {final_path}")
    
    return model, metrics
