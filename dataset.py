import os
import gc
import numpy as np
from sgfmill import sgf, boards


CHUNK_SIZE = 500      # SGF files per chunk; lower if you still OOM
FOLDER     = "games/9d"
OUT_DIR    = "go_chunks"


def board_to_tensor(board, current_player):
    size = board.side
    tensor = np.zeros((3, size, size), dtype=np.float32)
    for row in range(size):
        for col in range(size):
            color = board.get(row, col)
            if color == 'b':
                tensor[0, row, col] = 1.0
            elif color == 'w':
                tensor[1, row, col] = 1.0
    tensor[2, :, :] = 1.0 if current_player == 'b' else 0.0
    return tensor


def parse_game(path):
    with open(path, "rb") as f:
        game = sgf.Sgf_game.from_bytes(f.read())

    board      = boards.Board(game.get_size())
    winner     = game.get_winner()   # 'b', 'w', or None
    next_color = 'b'
    positions, next_moves, winners = [], [], []

    for node in game.get_main_sequence()[1:]:
        color, move = node.get_move()
        if move is not None:
            board.play(move[0], move[1], color)

        positions.append(board_to_tensor(board, next_color))
        next_moves.append(1 if next_color == 'b' else 0)
        winners.append(1 if winner == 'b' else (0 if winner == 'w' else -1))

        next_color = 'w' if next_color == 'b' else 'b'

    return positions, next_moves, winners


def flush_chunk(positions, next_moves, winners, chunk_idx):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"chunk_{chunk_idx:04d}.npz")
    np.savez_compressed(
        out_path,
        positions  = np.array(positions,  dtype=np.float32),  # (N, 3, H, W)
        next_to_go = np.array(next_moves, dtype=np.int8),     # (N,)  1=black, 0=white
        winner     = np.array(winners,    dtype=np.int8),     # (N,)  1=black, 0=white, -1=unknown
    )
    print(f"chunk {chunk_idx:04d}: {len(positions):>7,} positions → {out_path}")


def run(folder, chunk_size):
    files = sorted(f for f in os.listdir(folder) if f.endswith(".sgf"))
    print(f"Found {len(files)} SGF files, chunk size = {chunk_size}")

    positions, next_moves, winners = [], [], []
    chunk_idx = 0
    skipped   = 0

    for game_num, filename in enumerate(files):
        try:
            p, n, w = parse_game(os.path.join(folder, filename))
            positions.extend(p)
            next_moves.extend(n)
            winners.extend(w)
        except Exception as e:
            print(f"  Skipping {filename}: {e}")
            skipped += 1
            continue

        if (game_num + 1) % chunk_size == 0:
            flush_chunk(positions, next_moves, winners, chunk_idx)
            chunk_idx += 1
            positions, next_moves, winners = [], [], []
            gc.collect()

    if positions:
        flush_chunk(positions, next_moves, winners, chunk_idx)
        chunk_idx += 1

    print(f"\nDone. {chunk_idx} chunks written, {skipped} games skipped.")


if __name__ == "__main__":
    run(FOLDER, CHUNK_SIZE)