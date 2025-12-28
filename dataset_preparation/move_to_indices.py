import chess
import pandas as pd

# --- Step 1: Generate comprehensive UCI move list ---

def is_move_legal_in_any_position(move):
    # Try all piece types on from_square and see if move is legal
    for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
        board = chess.Board(None)
        board.set_piece_at(move.from_square, chess.Piece(pt, chess.WHITE))
        # Add kings for legality
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        if move in board.legal_moves:
            return True
    return False

uci_moves = set()
for from_square in chess.SQUARES:
    for to_square in chess.SQUARES:
        if from_square == to_square:
            continue
        # Normal move
        move = chess.Move(from_square, to_square)
        uci_moves.add(move.uci())
        # Promotions
        if chess.square_rank(to_square) in [0, 7]:
            for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                promo_move = chess.Move(from_square, to_square, promotion=promo)
                uci_moves.add(promo_move.uci())

# Filter moves to only those legal in at least one position
filtered_moves = []
for uci in sorted(uci_moves):
    move = chess.Move.from_uci(uci)
    if is_move_legal_in_any_position(move):
        filtered_moves.append(uci)

# Add castling moves explicitly if not present
for cmove in ["e1g1", "e1c1", "e8g8", "e8c8"]:
    if cmove not in filtered_moves:
        filtered_moves.append(cmove)

filtered_moves = sorted(filtered_moves)
print(f"Number of moves in list: {len(filtered_moves)}")
# Optionally save this list for future use
with open("generated_uci_moves.txt", "w") as f:
    for move in filtered_moves:
        f.write(move + "\n")

# --- Step 2: Map UCI moves in your CSV to indices ---

uci_move_to_index = {uci: idx for idx, uci in enumerate(filtered_moves)}

input_csv = "positions_with_human_moves3.csv"
output_csv = "positions_with_move_index3.csv"

df = pd.read_csv(input_csv)
df['MoveIndex'] = df['MoveUCI'].map(lambda move: uci_move_to_index.get(move, -1))
df = df[df['MoveIndex'] != -1]  # Filter out moves not in our list

df[['GameID', 'FEN', 'MoveUCI', 'MoveIndex']].to_csv(output_csv, index=False)
print(f"Done! Saved {len(df)} positions with move indices to {output_csv}")
