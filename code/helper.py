import os
import sys
import time
import logging
import json
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from tqdm import tqdm

# Import board_manipulation functions
from board_manipulation import (
    extract, solve_queens, hash_board, expand_board_dataset,
    rotate_board_data, generate_training_states, save_stateless_dataset_json
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def process_puzzle_images(
    images_folder: str,
    output_path: str,
    augment: bool = False,
    gen_count: int = 0,
    rotate: bool = False,
    make_states: bool = False
) -> Dict[str, Any]:
    """
    Process a folder of Queens puzzle images into a standardized dataset.

    Args:
        images_folder: Absolute path to folder containing JPG images
        output_path: Complete path with filename for output JSON
        augment: Whether to generate new boards through mutation
        gen_count: How many boards to generate if augmenting
        rotate: Whether to create rotated variations
        make_states: Whether to generate training states

    Returns:
        Dict with statistics about the processing operation
    """
    start_time = time.time()
    stats = {
        "total_images": 0,
        "successful_extractions": 0,
        "failed_extractions": 0,
        "successful_solutions": 0,
        "failed_solutions": 0,
        "augmented_boards": 0,
        "rotated_boards": 0,
        "training_states": 0,
        "total_boards_in_output": 0,
        "processing_time_seconds": 0,
    }

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # Get all JPG files in the folder
    image_files = []
    for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
        image_files.extend(list(Path(images_folder).glob(f"*{ext}")))
    
    stats["total_images"] = len(image_files)
    logger.info(f"Found {stats['total_images']} image files in {images_folder}")

    if stats["total_images"] == 0:
        logger.warning(f"No image files found in {images_folder}")
        return stats

    # Process each image and extract board data
    seed_dataset = []
    
    for img_path in tqdm(image_files, desc="Extracting boards"):
        try:
            # Extract region tensor from image
            region_tensor = extract(str(img_path))
            
            # Solve the board to get queen positions
            positions, solution_board = solve_queens(region_tensor)
            
            if positions is not None and solution_board is not None:
                # Create board entry with proper metadata
                board_entry = {
                    "region": region_tensor,
                    "queen_positions": positions,
                    "solution_board": solution_board,
                    "filename": img_path.name,
                    "source": img_path.name,
                    "iteration": 0
                }
                seed_dataset.append(board_entry)
                stats["successful_solutions"] += 1
            else:
                logger.warning(f"No valid solution found for {img_path.name}")
                stats["failed_solutions"] += 1
            
            stats["successful_extractions"] += 1
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {str(e)}")
            stats["failed_extractions"] += 1
    
    logger.info(f"Successfully extracted and solved {stats['successful_solutions']} boards")
    
    # Apply data augmentation if requested
    dataset = []
    
    # Initialize dataset with properly formatted seed entries
    # This ensures consistent format even when skipping expand_board_dataset
    for seed in seed_dataset:
        seed_entry = {
            "region": seed["region"],
            "queen_positions": seed["queen_positions"],
            "solution_board": seed["solution_board"],
            "source": seed["source"],
            "iteration": 0
        }
        print(f'SEED NAME {seed['source']}')
        dataset.append(seed_entry)
    
    if augment and gen_count > 0:
        logger.info(f"Generating {gen_count} new boards through mutation")
        generated_dataset, offspring_counter = expand_board_dataset(seed_dataset, target_size=gen_count)
        dataset.extend(generated_dataset)
        stats["augmented_boards"] = len(generated_dataset)
        logger.info(f"Generated {stats['augmented_boards']} new boards")

    # Apply rotational augmentation if requested
    if rotate:
        rotated_dataset = []
        for board in tqdm(dataset, desc="Creating rotations"):
            rotations = rotate_board_data(board)
            rotated_dataset.extend(rotations)
        
        dataset = rotated_dataset
        stats["rotated_boards"] = len(dataset) - stats["augmented_boards"] - len(seed_dataset)
        logger.info(f"Created {stats['rotated_boards']} rotated variations")
    
    # Generate training states if requested
    if make_states:
        states_dataset = []
        for board in tqdm(dataset, desc="Generating training states"):
            states = generate_training_states(board)
            states_dataset.extend(states)
        
        dataset = states_dataset
        stats["training_states"] = len(dataset)
        logger.info(f"Generated {stats['training_states']} training states")
        
        # Save dataset with states
        save_training_states_to_json(dataset, output_path)
    else:
        # Save regular dataset (no states)
        save_stateless_dataset_json(dataset, output_path)
    
    stats["total_boards_in_output"] = len(dataset)
    stats["processing_time_seconds"] = time.time() - start_time
    
    logger.info(f"Data processing complete. Output saved to {output_path}")
    logger.info(f"Total processing time: {stats['processing_time_seconds']:.2f} seconds")
    
    return stats

def save_training_states_to_json(dataset, filename):
    """
    Save training states dataset to JSON.
    Modified version of save_state_dataset_to_json for better integration.
    """
    serializable_data = []

    for entry in dataset:
        serializable_entry = {
            "region": entry["region"].tolist() if isinstance(entry["region"], np.ndarray) else entry["region"],
            "partial_board": entry["partial_board"].tolist() if isinstance(entry["partial_board"], np.ndarray) else entry["partial_board"],
            "label_board": entry["label_board"].tolist() if isinstance(entry["label_board"], np.ndarray) else entry["label_board"],
            "step": entry["step"],
            "source": entry.get("source", "unknown"),
            "iteration": entry.get("iteration", 0)
        }
        serializable_data.append(serializable_entry)

    with open(filename, "w") as f:
        json.dump(serializable_data, f)

    logger.info(f"Saved {len(serializable_data)} training state examples to {filename}")