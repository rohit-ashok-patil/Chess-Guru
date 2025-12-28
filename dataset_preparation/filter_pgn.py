import chess.pgn
import multiprocessing as mp
from tqdm import tqdm
import time
import os

def filter_game(game):
    # Filter by time control (greater than 10 minutes)
    time_control = game.headers.get('TimeControl', '')
    if time_control == '' or time_control == '?':
        return False
    try:
        base_time = int(time_control.split('+')[0])
    except Exception:
        return False
    if base_time < 600:  # less than 10 minutes
        return False

    # Filter by player ratings
    try:
        white_elo = int(game.headers.get('WhiteElo', '0'))
        black_elo = int(game.headers.get('BlackElo', '0'))
    except Exception:
        return False
    if white_elo < 1600 or black_elo < 1600:
        return False

    # Filter by number of moves (at least 50 full moves)
    node = game
    move_count = 0
    while node.variations:
        node = node.variations[0]
        move_count += 1
    if move_count < 50:
        return False

    return True

def find_game_offsets(pgn_file):
    """Find the file offsets of all games in a PGN file"""
    offsets = []
    with open(pgn_file, 'r', encoding='utf-8') as f:
        while True:
            offset = f.tell()
            headers = chess.pgn.read_headers(f)
            if headers is None:
                break
            offsets.append(offset)
            
            # Skip to the next game
            while True:
                line = f.readline()
                if not line or line.strip() == '' or line.startswith('['):
                    break
    
    return offsets

def process_games_chunk(args):
    """Process a chunk of games based on file offsets"""
    pgn_file, offsets, output_file = args
    filtered_games = []
    
    with open(pgn_file, 'r', encoding='utf-8') as f:
        for offset in offsets:
            f.seek(offset)
            game = chess.pgn.read_game(f)
            if game and filter_game(game):
                filtered_games.append(str(game))
    
    # Write filtered games to a temporary output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for game in filtered_games:
            f.write(game + "\n\n")
    
    return len(filtered_games)

def filter_pgn_parallel_optimized(input_pgn_path, output_pgn_path, num_processes=None):
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Create a temporary directory for chunk outputs
    temp_dir = "temp_pgn_chunks"
    os.makedirs(temp_dir, exist_ok=True)
    
    print("Finding game offsets...")
    offsets = find_game_offsets(input_pgn_path)
    total_games = len(offsets)
    print(f"Found {total_games} games in the PGN file")
    
    # Split offsets into chunks
    chunk_size = len(offsets) // num_processes
    offset_chunks = [offsets[i:i+chunk_size] for i in range(0, len(offsets), chunk_size)]
    
    # Create arguments for each process
    temp_output_files = [os.path.join(temp_dir, f"chunk_{i}.pgn") for i in range(len(offset_chunks))]
    process_args = [(input_pgn_path, chunk, output_file) 
                    for chunk, output_file in zip(offset_chunks, temp_output_files)]
    
    # Process chunks in parallel
    print(f"Processing {len(offset_chunks)} chunks in parallel...")
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_games_chunk, process_args),
            total=len(offset_chunks),
            desc="Processing chunks"
        ))
    
    # Combine results
    filtered_count = sum(results)
    print(f"Combining {filtered_count} filtered games...")
    
    with open(output_pgn_path, 'w', encoding='utf-8') as outfile:
        for temp_file in temp_output_files:
            if os.path.exists(temp_file):
                with open(temp_file, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                os.remove(temp_file)
    
    # Clean up
    os.rmdir(temp_dir)
    
    print(f"Done! Processed {total_games} games, filtered {filtered_count} games to {output_pgn_path}")

# Run the optimized parallel version
if __name__ == "__main__":
    start_time = time.time()
    filter_pgn_parallel_optimized('lichess_db_standard_rated_2017-03.pgn', 'filtered_output_parallel.pgn')
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.2f} seconds")
