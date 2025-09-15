#!/usr/bin/env python3
"""
Quick training script to create a chess model and export it to ONNX
This creates a minimal training run to generate a working model
"""
import os
import sys
import torch
import torch.nn as nn
import chess
import random

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from nmn_replacement import YatNMN

class NN1(nn.Module):
    """The neural network architecture for evaluating chess positions."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(13, 64)  # 12 pieces + 1 empty square
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=16, batch_first=True)
        self.neu = 512
        
        self.neurons = nn.Sequential(
            nn.Linear(64 * 64, self.neu, bias=False),
            YatNMN(self.neu, self.neu, bias=False),
            nn.Linear(self.neu, 64, bias=False),
            YatNMN(64, 64, bias=False),
            nn.Linear(64, 4, bias=False)
        )

    def forward(self, x):
        x = self.embedding(x)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.contiguous()
        x = x.view(x.size(0), -1)  # Flatten
        x = self.neurons(x)
        return x
    
    def export_to_onnx(self, onnx_path, device='cpu'):
        """Export the model to ONNX format for use with ONNX.js"""
        self.eval()
        self.to(device)
        
        # Create dummy input
        dummy_input = torch.randint(0, 13, (1, 64), dtype=torch.long).to(device)
        
        # Export to ONNX
        torch.onnx.export(
            self,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['board_input'],
            output_names=['evaluation'],
            dynamic_axes={
                'board_input': {0: 'batch_size'},
                'evaluation': {0: 'batch_size'}
            }
        )
        print(f"Model exported to ONNX format: {onnx_path}")
        return onnx_path

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Converts a chess board object to a tensor representation."""
    piece_encoding = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
    }
    tensor = torch.zeros(64, dtype=torch.long)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        tensor[square] = piece_encoding.get(piece.symbol(), 0) if piece else 0
    return tensor.unsqueeze(0)

def generate_training_data(num_positions=100):
    """Generate simple training data from random positions"""
    data = []
    
    for _ in range(num_positions):
        board = chess.Board()
        
        # Make some random moves to get different positions
        num_moves = random.randint(5, 20)
        for _ in range(num_moves):
            moves = list(board.legal_moves)
            if not moves:
                break
            move = random.choice(moves)
            board.push(move)
        
        # Simple evaluation based on material
        evaluation = simple_evaluate(board)
        tensor = board_to_tensor(board)
        
        data.append((tensor, evaluation))
    
    return data

def simple_evaluate(board):
    """Simple evaluation function for training data"""
    piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}
    
    if board.is_checkmate():
        return -100 if board.turn == chess.WHITE else 100
    
    if board.is_stalemate():
        return 0
    
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values.get(piece.symbol().lower(), 0)
            score += value if piece.color == chess.WHITE else -value
    
    return score

def quick_train():
    """Quick training run to create a basic model"""
    print("Creating and training a basic chess model...")
    
    # Create model
    model = NN1()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Generate training data
    print("Generating training data...")
    training_data = generate_training_data(200)
    
    # Quick training
    print("Training model...")
    model.train()
    for epoch in range(5):
        total_loss = 0
        for tensor, target in training_data:
            optimizer.zero_grad()
            output = model(tensor)[0][0]  # Get first output
            target_tensor = torch.tensor(target, dtype=torch.float32)
            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/5, Average Loss: {total_loss/len(training_data):.4f}")
    
    # Save PyTorch model
    torch.save(model.state_dict(), 'chessy_model.pth')
    print("Saved PyTorch model: chessy_model.pth")
    
    # Export to ONNX
    model.eval()
    onnx_path = 'chessy_model.onnx'
    model.export_to_onnx(onnx_path)
    print(f"Exported ONNX model: {onnx_path}")
    
    # Copy to web directory
    import shutil
    web_onnx_path = 'web/chessy_model.onnx'
    shutil.copy(onnx_path, web_onnx_path)
    print(f"Copied to web directory: {web_onnx_path}")
    
    return True

if __name__ == "__main__":
    quick_train()