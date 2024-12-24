# Libraries
import os
import json
from glob import glob
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Helper functions
from libs.openai import complete, embed, ENGINE
from libs.utils import load_dict_from_json, save_dict_as_json


def map_result(completion, results_map):
    """
    Maps a completion result to the desired output based on the provided results map.
        results_map = [
            ('yes', True),
            ...
        ]
    """
    # Clean and lowercase
    completion = completion.lower().strip().replace('"', '').replace("'", "").strip()

    # Check for exact match
    for result in results_map:
        if completion.startswith(result[0].lower()):
            return result[1]
        
    # If no exact match, return None
    return None


def process_task(task, results_map=None, model_large=False, json_mode=False, temperature=0.0, max_retries=3):
    """
    Processes a single task and maps its result based on the provided results map.
    """
    key, prompt = task

    # Compose and complete prompt
    for attempt in range(max_retries):

        # Max tokens
        max_tokens = 2048 if results_map is None else 16

        # Get completion
        completion = complete(prompt, max_tokens=max_tokens, model_large=model_large, json_mode=json_mode, temperature=temperature)
        if completion is None:
            print(f"Failed to get completion (attempt {attempt + 1}/{max_retries})")
            continue

        # If open-ended, return completion
        if results_map is None:
            return (key, completion)

        # Validate response
        result = map_result(completion, results_map)
        if result is None:
            print(f'Invalid completion "{completion}", (attempt {attempt + 1}/{max_retries})')
            continue

        # Return
        return (key, result)
    
    # If we've exhausted all retries
    print("All retry attempts failed.")
    return None


def process_batch(batch, filepath, results_map=None, model_large=False, json_mode=False, temperature=0.0):
    """
    Processes a batch of tasks and maps their results using the provided results map.
    """
    # Temporary batch results
    batch_narratives_dict = {}
    batch_uid = random.randint(1000000, 9000000)
    filepath_batch = filepath.replace('.json', f'_temp_{batch_uid}.json')

    # Iterate through the batch
    for task in tqdm(batch, desc=f'Batch {batch_uid}'):

        # Process task
        result = process_task(task, results_map=results_map, model_large=model_large, json_mode=json_mode, temperature=temperature)
        if result is None: continue

        # Store result
        key, label = result
        batch_narratives_dict[key] = label

        # Save annotations
        save_dict_as_json(batch_narratives_dict, filepath_batch)


def chunk_list(lst, n):
    """Split list into n roughly equal chunks"""
    avg = len(lst) // n
    remainder = len(lst) % n
    chunks = []
    start = 0

    for i in range(n):
        end = start + avg + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end

    return chunks


def combine_batches(filepath):

    # Glob all temp files
    filepaths_batches = glob(filepath.replace('.json', '_temp_*.json'))
    if len(filepaths_batches) == 0: 
        print('No temporary files found.')
        return

    # Load main file
    results_dict = {}
    if os.path.exists(filepath):
        results_dict = load_dict_from_json(filepath)

    # Count initial embeddings
    initial_count = len(results_dict)

    # Add temp files
    failed_to_combine = False
    for i, temp_filepath in enumerate(filepaths_batches):

        # Load batch file
        try:
            temp_dict = load_dict_from_json(temp_filepath)
        except:
            print(f'Error loading {temp_filepath}')
            failed_to_combine = True
            continue

        # Update the dictionary
        results_dict.update(temp_dict)
        new_count = len(results_dict)

        # Print progress
        print(f'[{i+1}/{len(filepaths_batches)}] Added {new_count - initial_count} new narratives.')

    # Save the updated dictionary
    save_dict_as_json(results_dict, filepath)

    # Delete temp files
    if failed_to_combine == False:
        for filepath in filepaths_batches:
            os.remove(filepath)


def parallel_process_batches(tasks, filepath, results_map=None, model_large=False, json_mode=False, temperature=0.0, num_workers=10):
    """
    Processes tasks in parallel and maps results using the provided results map.
    """
    if len(tasks) == 0:
        print('No tasks to process.')
        return

    # If ENGINE == 'local', num_workers = 1
    if ENGINE == 'local':
        num_workers = 1
        print('WARNING: ENGINE is set to "local". Setting num_workers to 1.')

    # Split tasks into chunks
    chunks = chunk_list(tasks, num_workers)
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    # If filepath doesn't exist, create it
    if not os.path.exists(filepath):
        save_dict_as_json({}, filepath)

    try:
        # Submit all chunks for processing
        futures = [executor.submit(process_batch, chunk, filepath, results_map, model_large, json_mode, temperature) for chunk in chunks]

        # Use tqdm to track progress
        for future in futures:
            future.result()  # Wait for each batch to finish

    except KeyboardInterrupt:
        print("\nInterrupt received. Attempting to finalize results...")
        # Force terminate all processes
        for pid, process in executor._processes.items():
            process.terminate()
        executor._processes.clear()
        executor.shutdown(wait=False)

    finally:
        # Combine results even if interrupted
        combine_batches(filepath)
        print("All temporary results combined into the main file.")


"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

def embed_task(task, embeddings_dict, max_retries=3):
    key, text = task

    # Skip if already processed
    if key in embeddings_dict and embeddings_dict[key] is not None:
        return None

    # Compose and complete prompt
    for attempt in range(max_retries):

        embedding = embed(text)
        if embedding is None:
            print(f"Failed to get embedding (attempt {attempt + 1}/{max_retries})")
            continue

        # Validate response
        if not isinstance(embedding, list):
            print(f'Invalid completion (attempt {attempt + 1}/{max_retries})')
            continue

        # Return
        return (key, embedding)
    
    # If we've exhausted all retries
    print("All retry attempts failed.")
    return None


def embed_batch(batch, filepath):

    # Temporary batch results
    batch_narratives_dict = {}
    batch_uid = random.randint(1000000, 9000000)
    filepath_batch = filepath.replace('.json', f'_temp_{batch_uid}.json')

    # Load annotations
    with open(filepath, 'r') as f:
        embeddings_dict = json.load(f)

    # Iterate through the batch
    for task in tqdm(batch, desc=f'Batch {batch_uid}'):

        # Process task
        result = embed_task(task, embeddings_dict)
        if result is None: continue

        # Store result
        key, label = result
        batch_narratives_dict[key] = label

        # Save annotations
        save_dict_as_json(batch_narratives_dict, filepath_batch)


def combine_embedded_batches(filepath):

    # Glob all temp files
    filepaths_batches = glob(filepath.replace('.json', '_temp_*.json'))
    if len(filepaths_batches) == 0: 
        print('No temporary files found.')
        return

    # Load main file
    embeddings_dict = {}
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            embeddings_dict = json.load(f)

    # Count initial embeddings
    initial_count = len(embeddings_dict)

    # Add temp files
    for i, temp_filepath in enumerate(filepaths_batches):

        # Load batch file
        try:
            with open(temp_filepath, 'r') as f:
                temp_dict = json.load(f)
        except:
            print(f'Error loading {temp_filepath}')
            continue
        
        # Update the dictionary
        embeddings_dict.update(temp_dict)
        new_count = len(embeddings_dict)

        # Print progress
        print(f'[{i+1}/{len(filepaths_batches)}] Added {new_count - initial_count} new narratives.')

    # Save the updated dictionary
    with open(filepath, 'w') as f:
        json.dump(embeddings_dict, f, indent=4, ensure_ascii=False, sort_keys=True)

    # Delete temp files
    for filepath in filepaths_batches:
        os.remove(filepath)


def parallel_embed_batches(tasks, filepath, num_workers=10):
    if len(tasks) == 0:
        print('No tasks to process.')
        return

    # Split tasks into chunks
    chunks = chunk_list(tasks, num_workers)
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    # If filepath doesn't exist, create it
    if not os.path.exists(filepath):
        save_dict_as_json({}, filepath)

    try:
        # Submit all chunks for processing
        futures = [executor.submit(embed_batch, chunk, filepath) for chunk in chunks]

        # Use tqdm to track progress
        for future in futures:
            future.result()  # Wait for each batch to finish

    except KeyboardInterrupt:
        print("\nInterrupt received. Attempting to finalize results...")
        # Force terminate all processes
        for pid, process in executor._processes.items():
            process.terminate()
        executor._processes.clear()
        executor.shutdown(wait=False)

    finally:
        # Combine results even if interrupted
        combine_embedded_batches(filepath)
        print("All temporary results combined into the main file.")
