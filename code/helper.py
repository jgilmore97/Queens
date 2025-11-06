import os
from pathlib import Path
from tqdm import tqdm
from board_manipulation import (
    extract,
    solve_queens,
    expand_board_dataset,
    rotate_board_data,
    generate_training_states,
    save_state_dataset_to_json,
    save_stateless_dataset_json
)


def process_puzzle_images(
    image_folder: str,
    destination: str,
    augment: bool = False,
    gen_count: int = 0,
    rotate: bool = False,
    include_states: bool = True,
    state_0_only: bool = False
):
    """
    Process Queens puzzle images into training/test data.
    
    Args:
        image_folder: Path to folder containing puzzle images (.jpg, .png)
        destination: Path where final JSON will be saved
        augment: If True, generate new boards via mutation using images as seeds
        gen_count: Number of boards to generate if augment=True (ignored if augment=False)
        rotate: If True, create 4 rotational variants of each board
        include_states: If True, create state sequences; if False, just board/solution
        state_0_only: If True and include_states=True, only create state 0 (empty board)
    
    Returns:
        None (saves JSON to destination)
    
    Example:
        # Create test set from images (no augmentation, just extract and solve)
        process_puzzle_images(
            image_folder="test_puzzles/",
            destination="test_set.json",
            augment=False,
            rotate=False,
            include_states=False
        )
        
        # Create training set with full augmentation
        process_puzzle_images(
            image_folder="seed_puzzles/",
            destination="training_set_with_states.json",
            augment=True,
            gen_count=5000,
            rotate=True,
            include_states=True,
            state_0_only=False
        )
        
        # Create state-0 fine-tuning dataset
        process_puzzle_images(
            image_folder="seed_puzzles/",
            destination="state0_training.json",
            augment=True,
            gen_count=10000,
            rotate=True,
            include_states=True,
            state_0_only=True
        )
    """
    
    # Validate inputs
    image_folder = Path(image_folder)
    if not image_folder.exists():
        raise ValueError(f"Image folder does not exist: {image_folder}")
    
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Queens Puzzle Image Processing Pipeline")
    print("="*60)
    print(f"Input: {image_folder}")
    print(f"Output: {destination}")
    print(f"Augment: {augment} (count={gen_count if augment else 'N/A'})")
    print(f"Rotate: {rotate}")
    print(f"States: {include_states} (state_0_only={state_0_only if include_states else 'N/A'})")
    print("="*60)
    
    print("\nExecution plan:")
    print(f"Extract tensors: [X]")
    print(f"Solve puzzles: [X]")
    print(f"Generate new puzzles: [{'X' if augment and gen_count > 0 else ' '}]")
    print(f"Rotate boards: [{'X' if rotate else ' '}]")
    print(f"Make states: [{'X' if include_states else ' '}]")
    print("="*60)
    
    # STEP 1: Extract and solve all images
    print("\n" + "="*60)
    print("Step 1/5: Extract boards from images")
    print("="*60)
    image_files = [f for f in image_folder.iterdir()]
    
    print(f"Found {len(image_files)} images")
    
    seed_dataset = []
    failed_extractions = []
    unsolvable_boards = []
    
    for img_path in tqdm(image_files, desc="Processing", unit="image"):
        try:
            region_board = extract(str(img_path), threshold=30)
            positions, solution_board = solve_queens(region_board)
            
            if positions is None:
                tqdm.write(f"Unsolvable: {img_path.name}")
                unsolvable_boards.append(img_path.name)
                continue
            
            seed_dataset.append({
                'region': region_board,
                'queen_positions': positions,
                'solution_board': solution_board,
                'source': img_path.name, 
                'iteration': 0
            })
            
        except Exception as e:
            tqdm.write(f"Failed: {img_path.name} - {e}")
            failed_extractions.append(img_path.name)
            print(f"\nTerminating due to extraction failure")
            raise
    
    print(f"Extracted and solved {len(seed_dataset)} boards")
    if unsolvable_boards:
        print(f"Skipped {len(unsolvable_boards)} unsolvable boards")
    
    # STEP 2: Augment via mutation (optional)
    print("\n" + "="*60)
    print(f"Step 2/5: {'Generate new boards' if augment and gen_count > 0 else 'Skip augmentation'}")
    print("="*60)
    
    if augment and gen_count > 0:
        print(f"Target: {gen_count} boards")
        generated_dataset, offspring_counter = expand_board_dataset(seed_dataset, target_size=gen_count)
        print(f"Generated {len(generated_dataset)} boards")
        all_boards = seed_dataset + generated_dataset
    else:
        all_boards = seed_dataset
    
    print(f"Total boards: {len(all_boards)}")
    
    # STEP 3: Rotate (optional)
    print("\n" + "="*60)
    print(f"Step 3/5: {'Rotate boards' if rotate else 'Skip rotation'}")
    print("="*60)
    
    if rotate:
        rotated_boards = []
        for board_data in tqdm(all_boards, desc="Rotating", unit="board"):
            rotated_variants = rotate_board_data(board_data)
            rotated_boards.extend(rotated_variants)
        all_boards = rotated_boards
    
    print(f"Total boards: {len(all_boards)}")
    
    # STEP 4: Generate states (optional)
    print("\n" + "="*60)
    print(f"Step 4/5: {'Generate states' if include_states else 'Skip states'}")
    print("="*60)
    
    if include_states:
        all_training_examples = []
        for board_data in tqdm(all_boards, desc="Generating states", unit="board"):
            states = generate_training_states(board_data, state_0_only=state_0_only)
            all_training_examples.extend(states)
        print(f"Generated {len(all_training_examples)} training examples")
    else:
        all_training_examples = None
    
    # STEP 5: Save to JSON
    print("\n" + "="*60)
    print("Step 5/5: Save to JSON")
    print("="*60)
    
    if include_states:
        save_state_dataset_to_json(all_training_examples, str(destination))
        final_count = len(all_training_examples)
    else:
        save_stateless_dataset_json(all_boards, str(destination))
        final_count = len(all_boards)
    
    print("\n" + "="*60)
    print("Processing complete")
    print("="*60)
    print(f"Output: {destination}")
    print(f"Total items: {final_count:,}")
    print("="*60)