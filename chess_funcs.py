import chess
import chess.pgn
import torch
from torch.utils.data import Dataset
import pandas as pd
import chess.svg
# import cairosvg
import pygame
import numpy as np
from io import BytesIO


def fen_to_tensor(fen):
    piece_dict = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
    }

    board = chess.Board(fen)
    board_tensor = torch.zeros((13, 8, 8), dtype=torch.float32)  # 13 channels, 8x8 board

    castle_dict = {'K': (7, 6), 'Q': (7, 2), 'k': (0, 6), 'q': (0, 2)}

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            index = piece_dict[piece.symbol()]
            board_tensor[index, row, col] = 1

    fen_split = fen.split(' ')
    active_player = 1 if fen_split[1] == 'w' else 0
    castle = fen_split[2]
    en_passant = fen_split[3]
    halfmove_clock = int(fen_split[4]) / 100.0

    if en_passant != '-':
        row = 8 - int(en_passant[1])
        col = ord(en_passant[0]) - 97
        board_tensor[12, row, col] = 1
    if castle != '-':
        for piece in castle:
            r, c = castle_dict[piece]
            board_tensor[12, r, c] = 1

    return board_tensor, active_player, halfmove_clock


class ChessDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.positions = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fen = self.positions.iloc[idx]['FEN']
        evaluation_str = self.positions.iloc[idx]['Evaluation']

        if evaluation_str.startswith('#'):
            if evaluation_str[1] == '-':
                evaluation = -10000.0
            else:
                evaluation = 10000.0
        else:
            evaluation = float(evaluation_str)

        board_tensor, active_player, halfmove_clock = fen_to_tensor(fen)
        sample = {
            'board_tensor': board_tensor,
            'active_player': active_player,
            'halfmove_clock': halfmove_clock,
            'evaluation': torch.tensor(evaluation, dtype=torch.float32)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class ChessAI:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode

    def evaluate_position(self, fen):
        board_tensor, active_player, halfmove_clock = fen_to_tensor(fen)

        board_tensor = board_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        active_player = torch.tensor([active_player], dtype=torch.long).to(self.device)
        halfmove_clock = torch.tensor([halfmove_clock], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            score = self.model(board_tensor, active_player, halfmove_clock).item()
        return score

    def suggest_move(self, fen):
        board = chess.Board(fen)
        best_move = None
        best_score = float('-inf') if board.turn == chess.WHITE else float('inf')

        for move in board.legal_moves:
            board.push(move)
            score = self.evaluate_position(board.fen())
            board.pop()

            if board.turn == chess.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
        return best_move.uci() if best_move else None

    def render_board(self, board, screen):
        svg_board = chess.svg.board(board=board)
        png_data = cairosvg.svg2png(bytestring=svg_board)
        image = pygame.image.load(BytesIO(png_data))
        image = pygame.transform.scale(image, (600, 600))
        screen.blit(image, (0, 0))
        pygame.display.flip()

    def inline_game(self):
        print("Welcome to SimpleChessAI!")
        user_color = input("Choose your color (white/black): ").strip().lower()
        if user_color not in ['white', 'black']:
            print("Invalid color. Defaulting to white.")
            user_color = 'white'

        user_is_white = (user_color == 'white')
        board = chess.Board()

        while not board.is_game_over():
            print("\nCurrent board:")
            print(board)

            if (board.turn == chess.WHITE and user_is_white) or (board.turn == chess.BLACK and not user_is_white):
                print("Your turn!")
                move = None
                while move not in board.legal_moves:
                    user_move = input("Enter your move in UCI format (e.g., e2e4): ").strip()
                    try:
                        move = chess.Move.from_uci(user_move)
                        if move not in board.legal_moves:
                            print("Illegal move. Try again.")
                    except ValueError:
                        print("Invalid move format. Try again.")
                board.push(move)
            else:
                ai_move = self.suggest_move(board.fen())
                print(f"AI plays: {ai_move}")
                board.push(chess.Move.from_uci(ai_move))

        print("\nGame over!")
        print(board)
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                print("Black wins by checkmate!")
            else:
                print("White wins by checkmate!")
        elif board.is_stalemate():
            print("It's a stalemate!")
        elif board.is_insufficient_material():
            print("Draw due to insufficient material!")
        elif board.is_seventyfive_moves():
            print("Draw due to the seventy-five-move rule!")
        elif board.is_fivefold_repetition():
            print("Draw due to fivefold repetition!")

    '''def inline_game(self):
        print("Welcome to SimpleChessAI!")
        user_color = input("Choose your color (white/black): ").strip().lower()
        if user_color not in ['white', 'black']:
            print("Invalid color. Defaulting to white.")
            user_color = 'white'

        user_is_white = (user_color == 'white')
        board = chess.Board()

        pygame.init()
        screen = pygame.display.set_mode((600, 600))
        pygame.display.set_caption("SimpleChessAI Game")

        running = True
        while not board.is_game_over() and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.render_board(board, screen)

            if (board.turn == chess.WHITE and user_is_white) or (board.turn == chess.BLACK and not user_is_white):
                print("Your turn!")
                move = None
                while move not in board.legal_moves:
                    user_move = input("Enter your move in UCI format (e.g., e2e4): ").strip()
                    try:
                        move = chess.Move.from_uci(user_move)
                        if move not in board.legal_moves:
                            print("Illegal move. Try again.")
                    except ValueError:
                        print("Invalid move format. Try again.")
                board.push(move)
            else:
                ai_move = self.suggest_move(board.fen())
                print(f"AI plays: {ai_move}")
                board.push(chess.Move.from_uci(ai_move))

        self.render_board(board, screen)

        print("\nGame over!")
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                print("Black wins by checkmate!")
            else:
                print("White wins by checkmate!")
        elif board.is_stalemate():
            print("It's a stalemate!")
        elif board.is_insufficient_material():
            print("Draw due to insufficient material!")
        elif board.is_seventyfive_moves():
            print("Draw due to the seventy-five-move rule!")
        elif board.is_fivefold_repetition():
            print("Draw due to fivefold repetition!")

        pygame.quit()'''


class RandomChessAi:
    def __init__(self):
        pass

    def suggest_move(self, fen):
        board = chess.Board(fen)
        move = np.random.choice(list(board.legal_moves)).uci()
        return move


def chess_game(model1, model2):
    board = chess.Board()
    pgn = ""
    i = 1
    while True:
        pgn += str(i) + "."
        move1 = model1.suggest_move(board.fen())
        pgn += move1 + " "
        board.push(chess.Move.from_uci(move1))
        if board.outcome():
            if board.is_checkmate():
                return 'White'
            else:
                return "Draw"

        move2 = model2.suggest_move(board.fen())
        pgn += move2 + " "
        board.push(chess.Move.from_uci(move2))
        if board.outcome():
            if board.is_checkmate():
                return 'Black'
            else:
                return "Draw"
        i += 1


if __name__ == '__main__':
    model = torch.load("model.pn", weights_only=False)
    chess_ai = ChessAI(model)
    random_chess_ai = RandomChessAi()
    data = pd.Series(index=list(range(1)), dtype=str)
    for i in range(1):
        data[i] = chess_game(chess_ai, chess_ai)
    print(data.value_counts())
