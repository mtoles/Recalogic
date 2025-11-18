#!/usr/bin/env python3
"""
SBERT Training Script for Feature-Distance Dataset

Trains a Sentence-BERT model using standard contrastive learning on the feature-distance dataset.
Ignores the existing distance metric and uses MultipleNegativesRankingLoss for contrastive learning.
"""

import argparse
import json
import logging
import os
from typing import List, Dict, Tuple
import random

import torch
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
)
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_jsonl_dataset(filepath: str) -> List[Dict]:
    """Load JSONL dataset from file."""
    logger.info(f"Loading dataset from {filepath}")
    dataset = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    logger.info(f"Loaded {len(dataset)} examples")
    return dataset


def create_contrastive_examples(dataset: List[Dict], eval_condition: str) -> List[InputExample]:
    """
    Create contrastive training examples from the dataset.
    
    Each example in the dataset has:
    - nl_query: natural language query
    - unique_pos_features: features that match positively
    - unique_neg_features: features that don't match
    - common_features: features common to both
    
    We create positive pairs (query, positive_features) and negative pairs (query, negative_features).
    For contrastive learning, we'll use triplets: (query, positive_text, negative_text)
    """
    eval_condition_map = {
        "ours": "nl_query",
        "baseline": "original_query"
    }
    examples = []
    
    for idx, item in enumerate(dataset):
        # query = item.get('nl_query', '')
        query = eval_condition_map[eval_condition]
        assert query
        
        # Get feature lists
        unique_pos = item.get('unique_pos_features', [])
        unique_neg = item.get('unique_neg_features', [])
        common = item.get('common_features', [])
        neither = item.get('neither_features', [])
        
        # Create positive text (features that should match the query)
        positive_features = unique_pos + common
        if positive_features:
            positive_text = ", ".join(positive_features)
        else:
            # Fallback to item description if available
            positive_text = item.get('item', '')
        
        # Create negative text (features that should NOT match the query)
        negative_features = unique_neg + neither[:3]  # Include some neither features as negatives
        if negative_features:
            negative_text = ", ".join(negative_features)
        else:
            # Use a generic negative if none available
            negative_text = "unrelated product features"
        
        # Create triplet: InputExample with label as similarity score
        # For positive pairs, use label 1.0
        if positive_text:
            examples.append(InputExample(texts=[query, positive_text], label=1.0))
        
        # For negative pairs, use label 0.0
        if negative_text:
            examples.append(InputExample(texts=[query, negative_text], label=0.0))
    
    logger.info(f"Created {len(examples)} training examples")
    return examples


def create_triplet_examples(dataset: List[Dict]) -> List[InputExample]:
    """
    Create triplet training examples (anchor, positive, negative).
    This is used for TripletLoss.
    """
    examples = []
    
    for idx, item in enumerate(dataset):
        query = item.get('nl_query', '')
        if not query:
            continue
        
        unique_pos = item.get('unique_pos_features', [])
        unique_neg = item.get('unique_neg_features', [])
        common = item.get('common_features', [])
        neither = item.get('neither_features', [])
        
        # Create positive text
        positive_features = unique_pos + common
        if positive_features:
            positive_text = ", ".join(positive_features)
        else:
            continue
        
        # Create negative text
        negative_features = unique_neg + neither[:3]
        if negative_features:
            negative_text = ", ".join(negative_features)
        else:
            continue
        
        # Create triplet: (anchor=query, positive=pos_features, negative=neg_features)
        examples.append(InputExample(texts=[query, positive_text, negative_text]))
    
    logger.info(f"Created {len(examples)} triplet examples")
    return examples


def train_sbert(
    dataset_path: str,
    eval_dataset_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    output_dir: str = "./models/sbert-contrastive",
    loss_type: str = "cosine",
    batch_size: int = 16,
    epochs: int = 4,
    warmup_steps: int = 100,
    evaluation_steps: int = 500,
    use_triplet: bool = False,
):
    """
    Train SBERT model using contrastive learning.
    
    Args:
        dataset_path: Path to JSONL dataset
        model_name: Base model to fine-tune
        output_dir: Directory to save trained model
        loss_type: Type of loss - 'cosine' (CosineSimilarityLoss) or 'mnrl' (MultipleNegativesRankingLoss) or 'triplet'
        batch_size: Training batch size
        epochs: Number of training epochs
        warmup_steps: Number of warmup steps
        evaluation_steps: Steps between evaluations
        use_triplet: Whether to use triplet loss
    """
    # Load dataset
    dataset = load_jsonl_dataset(dataset_path)
    
    # Create training examples
    if use_triplet or loss_type == 'triplet':
        train_examples = create_triplet_examples(dataset)
        loss_type = 'triplet'
    else:
        train_examples = create_contrastive_examples(dataset)
    
    # Don't shuffle because examples are grouped by topic, to prevent leakage
    # random.shuffle(train_examples)
    
    # Split into train/eval (80/20)
    split_idx = int(0.8 * len(train_examples))
    train_data = train_examples[:split_idx]
    eval_data = train_examples[split_idx:]
    
    logger.info(f"Training examples: {len(train_data)}")
    logger.info(f"Evaluation examples: {len(eval_data)}")
    
    # Load base model
    logger.info(f"Loading base model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Create data loader
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    
    # Choose loss function
    if loss_type == 'triplet':
        logger.info("Using TripletLoss for training")
        train_loss = losses.TripletLoss(model=model)
    elif loss_type == 'mnrl':
        logger.info("Using MultipleNegativesRankingLoss for training")
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    else:  # cosine similarity loss (default)
        logger.info("Using CosineSimilarityLoss for training")
        train_loss = losses.CosineSimilarityLoss(model=model)
    
    # Create evaluator if we have eval data
    evaluator = None
    if eval_data and not use_triplet:
        # Extract sentence pairs and scores for evaluation
        sentences1 = [ex.texts[0] for ex in eval_data]
        sentences2 = [ex.texts[1] for ex in eval_data]
        scores = [ex.label for ex in eval_data]
        evaluator = evaluation.EmbeddingSimilarityEvaluator(
            sentences1, sentences2, scores, name="eval"
        )
    
    # Training
    logger.info("Starting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=evaluation_steps,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        show_progress_bar=True,
    )
    
    logger.info(f"Training complete! Model saved to {output_dir}")
    
    # Save a sample embedding for verification
    logger.info("Testing model with sample query...")
    sample_query = dataset[0].get('nl_query', 'test query')
    embedding = model.encode(sample_query)
    logger.info(f"Sample embedding shape: {embedding.shape}")
    logger.info(f"Sample embedding (first 10 dims): {embedding[:10]}")
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train SBERT model on feature-distance dataset using contrastive learning"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/feature-distance-dataset_gemini-2.5-flash-lite_10000.jsonl",
        help="Path to JSONL dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Base Sentence-BERT model to fine-tune ",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/sbert-contrastive",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=['cosine', 'mnrl', 'triplet'],
        default="cosine",
        help="Loss function: cosine (CosineSimilarityLoss), mnrl (MultipleNegativesRankingLoss), or triplet (TripletLoss)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Steps between evaluations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_arguemtn(
        "--eval-dataset",
        type=str,
        default="ours",
        help="Path to evaluation dataset",
    )
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Train model
    train_sbert(
        dataset_path=args.dataset,
        eval_dataset_path=args.eval_dataset,
        model_name=args.model,
        output_dir=args.output,
        loss_type=args.loss,
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        evaluation_steps=args.eval_steps,
        use_triplet=(args.loss == 'triplet'),
    )


if __name__ == "__main__":
    main()

