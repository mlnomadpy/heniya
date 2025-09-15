#!/usr/bin/env python3
"""
Test script to verify the chess neural network can be created and exported to ONNX.
"""
import os
import sys
import torch
import chess

# Add current directory to path to import our modules
sys.path.insert(0, os.path.dirname(__file__))

from nmn_replacement import YatNMN
import torch.nn as nn

# Simplified version of NN1 for testing
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
            opset_version=13,  # Updated to version 13 for better compatibility
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

def test_model():
    print("Testing neural network model...")
    
    # Create model
    model = NN1()
    model.eval()
    
    # Test with a chess board
    board = chess.Board()
    tensor = board_to_tensor(board)
    
    print(f"Input tensor shape: {tensor.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(tensor)
    
    print(f"Output shape: {output.shape}")
    print(f"Output value: {output[0][0].item():.4f}")
    
    # Export to ONNX
    onnx_path = "test_model.onnx"
    model.export_to_onnx(onnx_path)
    
    # Verify ONNX file was created
    if os.path.exists(onnx_path):
        file_size = os.path.getsize(onnx_path)
        print(f"ONNX file created successfully: {onnx_path} ({file_size} bytes)")
        return True
    else:
        print("ERROR: ONNX file was not created")
        return False

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)