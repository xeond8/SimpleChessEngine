from chess_funcs import ChessAI
import torch

model = torch.load("model.pn", weights_only=False)
chess_ai = ChessAI(model)
chess_ai.inline_game()
