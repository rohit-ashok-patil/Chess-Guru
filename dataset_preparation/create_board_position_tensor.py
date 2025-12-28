import numpy as np
import chess
import pandas as pd
import os
import csv
from tqdm import tqdm

piece_to_plane = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def fen_to_maia2_tensor(
    fen,
    last_move_uci=None,
    move_count=0,
    fifty_move_count=0,
    repetition_count=0,
    repetition_threefold=False
):
    board = chess.Board(fen)
    tensor = np.zeros((8, 8, 60), dtype=np.float32)  # 60 planes

    # 12 planes for pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            plane = piece_to_plane[piece.symbol()]
            tensor[row, col, plane] = 1

    # Castling rights (4 planes: WK, WQ, BK, BQ) split with rook presence
    tensor[:, :, 12] = int(board.has_kingside_castling_rights(chess.WHITE) and board.piece_at(chess.H1) and board.piece_at(chess.H1).symbol() == 'R')
    tensor[:, :, 13] = int(board.has_queenside_castling_rights(chess.WHITE) and board.piece_at(chess.A1) and board.piece_at(chess.A1).symbol() == 'R')
    tensor[:, :, 14] = int(board.has_kingside_castling_rights(chess.BLACK) and board.piece_at(chess.H8) and board.piece_at(chess.H8).symbol() == 'r')
    tensor[:, :, 15] = int(board.has_queenside_castling_rights(chess.BLACK) and board.piece_at(chess.A8) and board.piece_at(chess.A8).symbol() == 'r')

    # Side to move (1 plane)
    tensor[:, :, 16] = float(board.turn == chess.WHITE)

    # En passant (1 plane)
    if board.ep_square is not None:
        row = 7 - (board.ep_square // 8)
        col = board.ep_square % 8
        tensor[row, col, 17] = 1

    # Attack maps (2 planes): squares attacked by white and black
    for sq in chess.SQUARES:
        row, col = 7 - (sq // 8), sq % 8
        if board.is_attacked_by(chess.WHITE, sq):
            tensor[row, col, 18] = 1
        if board.is_attacked_by(chess.BLACK, sq):
            tensor[row, col, 19] = 1

    # Defense maps (2 planes): squares defended by white and black
    for sq in chess.SQUARES:
        row, col = 7 - (sq // 8), sq % 8
        piece = board.piece_at(sq)
        if piece:
            if piece.color == chess.WHITE:
                tensor[row, col, 20] = 1
            else:
                tensor[row, col, 21] = 1

    # Mobility planes (2 planes): number of legal moves for white and black pieces
    white_mobility = np.zeros((8, 8), dtype=np.float32)
    black_mobility = np.zeros((8, 8), dtype=np.float32)
    for move in board.legal_moves:
        from_sq = move.from_square
        row = 7 - (from_sq // 8)
        col = from_sq % 8
        piece = board.piece_at(from_sq)
        if piece:
            if piece.color == chess.WHITE:
                white_mobility[row, col] += 1
            else:
                black_mobility[row, col] += 1
    tensor[:, :, 22] = white_mobility / 27.0  # Normalize
    tensor[:, :, 23] = black_mobility / 27.0

    # Pinned pieces (2 planes): white and black
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and board.is_pinned(piece.color, sq):
            row, col = 7 - (sq // 8), sq % 8
            if piece.color == chess.WHITE:
                tensor[row, col, 24] = 1
            else:
                tensor[row, col, 25] = 1

    # Pieces under threat (2 planes): white and black
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and board.is_attacked_by(not piece.color, sq):
            row, col = 7 - (sq // 8), sq % 8
            if piece.color == chess.WHITE:
                tensor[row, col, 26] = 1
            else:
                tensor[row, col, 27] = 1

    # Pawn structure planes (6 planes): passed, isolated, doubled pawns for white and black
    def pawn_structure_planes(color):
        passed = np.zeros((8, 8), dtype=np.float32)
        isolated = np.zeros((8, 8), dtype=np.float32)
        doubled = np.zeros((8, 8), dtype=np.float32)
        pawns = [sq for sq in chess.SQUARES if board.piece_at(sq) and board.piece_at(sq).symbol() == ('P' if color == chess.WHITE else 'p')]
        files = [sq % 8 for sq in pawns]
        file_counts = {f: files.count(f) for f in set(files)}
        for sq in pawns:
            row, col = 7 - (sq // 8), sq % 8
            # Passed pawn check
            is_passed = True
            for f in [col - 1, col, col + 1]:
                if 0 <= f <= 7:
                    for r in (range(row - 1, -1, -1) if color == chess.WHITE else range(row + 1, 8)):
                        sq_check = (7 - r) * 8 + f
                        piece = board.piece_at(sq_check)
                        if piece and piece.symbol().lower() == 'p' and piece.color != color:
                            is_passed = False
                            break
                    if not is_passed:
                        break
            if is_passed:
                passed[row, col] = 1
            # Isolated pawn check
            is_isolated = True
            for f in [col - 1, col + 1]:
                if 0 <= f <= 7 and file_counts.get(f, 0) > 0:
                    is_isolated = False
                    break
            if is_isolated:
                isolated[row, col] = 1
            # Doubled pawn check
            if file_counts.get(col, 0) > 1:
                doubled[row, col] = 1
        return passed, isolated, doubled

    wp_passed, wp_isolated, wp_doubled = pawn_structure_planes(chess.WHITE)
    bp_passed, bp_isolated, bp_doubled = pawn_structure_planes(chess.BLACK)
    tensor[:, :, 28] = wp_passed
    tensor[:, :, 29] = wp_isolated
    tensor[:, :, 30] = wp_doubled
    tensor[:, :, 31] = bp_passed
    tensor[:, :, 32] = bp_isolated
    tensor[:, :, 33] = bp_doubled

    # King safety planes (4 planes): white castled, black castled, white king in check, black king in check
    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)
    if white_king_sq is not None:
        row, col = 7 - (white_king_sq // 8), white_king_sq % 8
        tensor[row, col, 34] = int(board.has_kingside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.WHITE))
        tensor[row, col, 36] = int(board.is_check() and board.turn == chess.WHITE)
    if black_king_sq is not None:
        row, col = 7 - (black_king_sq // 8), black_king_sq % 8
        tensor[row, col, 35] = int(board.has_kingside_castling_rights(chess.BLACK) or board.has_queenside_castling_rights(chess.BLACK))
        tensor[row, col, 37] = int(board.is_check() and board.turn == chess.BLACK)

    # Move count (ply) normalized as a plane
    tensor[:, :, 38] = move_count / 200.0  # Assuming max 200 plies

    # Fifty-move rule count normalized
    tensor[:, :, 39] = fifty_move_count / 100.0  # Assuming max 100

    # Last move played (2 planes): from and to squares
    if last_move_uci:
        from_sq = chess.parse_square(last_move_uci[:2])
        to_sq = chess.parse_square(last_move_uci[2:4])
        tensor[7 - (from_sq // 8), from_sq % 8, 40] = 1
        tensor[7 - (to_sq // 8), to_sq % 8, 41] = 1

    # Material count planes (2 planes): white and black, normalized (max 39 for white, 39 for black)
    white_material = sum([piece.piece_type for piece in board.piece_map().values() if piece.color == chess.WHITE])
    black_material = sum([piece.piece_type for piece in board.piece_map().values() if piece.color == chess.BLACK])
    tensor[:, :, 42] = white_material / 39.0
    tensor[:, :, 43] = black_material / 39.0

    # Material imbalance (1 plane): white - black, normalized to [-1, 1]
    imbalance = (white_material - black_material) / 39.0
    tensor[:, :, 44] = imbalance

    # Bishop pair (2 planes): 1 if has bishop pair, else 0
    white_bishops = [piece for piece in board.piece_map().values() if piece.color == chess.WHITE and piece.piece_type == chess.BISHOP]
    black_bishops = [piece for piece in board.piece_map().values() if piece.color == chess.BLACK and piece.piece_type == chess.BISHOP]
    if len(white_bishops) >= 2:
        tensor[:, :, 45] = 1
    if len(black_bishops) >= 2:
        tensor[:, :, 46] = 1

    # Rook vs two minors (2 planes): 1 if side has rook vs two minors, else 0
    def rook_vs_two_minors(color):
        rooks = sum(1 for piece in board.piece_map().values() if piece.color == color and piece.piece_type == chess.ROOK)
        minors = sum(1 for piece in board.piece_map().values() if piece.color == color and piece.piece_type in [chess.BISHOP, chess.KNIGHT])
        return int(rooks == 1 and minors == 2)
    tensor[:, :, 47] = rook_vs_two_minors(chess.WHITE)
    tensor[:, :, 48] = rook_vs_two_minors(chess.BLACK)

    # Danger zone planes: squares where king would be in check if moved there (2 planes)
    for sq in chess.SQUARES:
        row, col = 7 - (sq // 8), sq % 8
        if board.is_attacked_by(chess.BLACK, sq):
            tensor[row, col, 49] = 1
        if board.is_attacked_by(chess.WHITE, sq):
            tensor[row, col, 50] = 1

    # Fork threat planes (knight/queen, both colors, 4 planes)
    def fork_threat_plane(color, piece_type):
        fork_plane = np.zeros((8, 8), dtype=np.float32)
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if not piece or piece.color != color or piece.piece_type != piece_type:
                continue
            for move in board.legal_moves:
                if move.from_square != sq:
                    continue
                temp_board = board.copy()
                temp_board.push(move)
                attacked = []
                for target_sq in chess.SQUARES:
                    if temp_board.is_attacked_by(color, target_sq):
                        target_piece = temp_board.piece_at(target_sq)
                        if target_piece and target_piece.color != color:
                            attacked.append(target_sq)
                if len(set(attacked)) >= 2:
                    row, col = 7 - (move.to_square // 8), move.to_square % 8
                    fork_plane[row, col] = 1
        return fork_plane

    tensor[:, :, 51] = fork_threat_plane(chess.WHITE, chess.KNIGHT)
    tensor[:, :, 52] = fork_threat_plane(chess.BLACK, chess.KNIGHT)
    tensor[:, :, 53] = fork_threat_plane(chess.WHITE, chess.QUEEN)
    tensor[:, :, 54] = fork_threat_plane(chess.BLACK, chess.QUEEN)

    # Promotion threat planes (2): pawns one step from promotion
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.PAWN:
            row, col = 7 - (sq // 8), sq % 8
            if piece.color == chess.WHITE and row == 1:
                tensor[row, col, 55] = 1
            if piece.color == chess.BLACK and row == 6:
                tensor[row, col, 56] = 1

    # Control over key squares (e4, d4, e5, d5) (2 planes: white/black control)
    key_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    for sq in key_squares:
        row, col = 7 - (sq // 8), sq % 8
        if board.is_attacked_by(chess.WHITE, sq):
            tensor[row, col, 57] = 1
        if board.is_attacked_by(chess.BLACK, sq):
            tensor[row, col, 58] = 1

    # Open files (1 plane): mark all squares on open files
    for file in range(8):
        file_squares = [8 * rank + file for rank in range(8)]
        has_white_pawn = any(board.piece_at(sq) and board.piece_at(sq).symbol() == 'P' for sq in file_squares)
        has_black_pawn = any(board.piece_at(sq) and board.piece_at(sq).symbol() == 'p' for sq in file_squares)
        if not has_white_pawn and not has_black_pawn:
            for sq in file_squares:
                row, col = 7 - (sq // 8), sq % 8
                tensor[row, col, 59] = 1

    # Threefold repetition status (1 plane, all squares filled)
    if repetition_threefold:
        tensor[:, :, 60] = 1

    return tensor
