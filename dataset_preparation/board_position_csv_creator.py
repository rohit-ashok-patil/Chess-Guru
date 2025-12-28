import chess.pgn
import csv
from tqdm import tqdm

input_pgn = "filtered_output_pgn_extract1.pgn"
output_csv = "positions_with_human_moves3.csv"

def extract_positions_and_moves(pgn_file, max_positions=None):
    positions = []
    game_id = 0
    with open(pgn_file, encoding="utf-8") as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                fen = board.fen()  # FEN before the move
                uci_move = move.uci()
                san_move = board.san(move)
                positions.append((game_id, fen, uci_move, san_move))
                board.push(move)
                if max_positions and len(positions) >= max_positions:
                    return positions
            game_id += 1
    return positions

if __name__ == "__main__":
    print("Extracting up to 1,00,000 positions and moves from PGN file...")
    positions = extract_positions_and_moves(input_pgn, max_positions=100000)
    print(f"Total positions extracted: {len(positions)}")

    # Write header and data
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["GameID", "FEN", "MoveUCI", "MoveSAN"])
        for row in tqdm(positions, desc="Writing to CSV", unit="pos"):
            csv_writer.writerow(row)

    print(f"\nExtraction complete. Data saved to {output_csv}")
