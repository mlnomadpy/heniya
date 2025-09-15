# main_chess.py

import os
import subprocess
import sys
from io import BytesIO

# ==============================================================================
# 1. DEPENDENCY INSTALLATION
# ==============================================================================
def install_dependencies():
    """Installs all required packages."""
    packages = [
        "torch",
        "python-chess",
        "gradio",
        "cairosvg",
        "Pillow"
    ]
    print("Installing required packages...")
    try:
        for package in packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("All packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        sys.exit(1)

# To install dependencies, uncomment and run this line in a cell by itself first.
# install_dependencies()


# ==============================================================================
# 2. IMPORTS
# ==============================================================================
import chess
import chess.engine as eng
import chess.svg
import gradio as gr
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from PIL import Image

# Import our replacement for YatNMN
try:
    from nmn.torch.nmn import YatNMN
except ImportError:
    # Use our local replacement
    from nmn_replacement import YatNMN

try:
    import cairosvg
except (OSError, ImportError):
    print("Warning: CairoSVG is not installed or configured correctly. Board rendering in the UI may fail.")
    print("For Windows, see: https://github.com/Kozea/WeasyPrint/blob/main/docs/install.rst#windows")
    print("For Linux, run: sudo apt-get install libcairo2-dev")
    cairosvg = None


# ==============================================================================
# 3. SHARED COMPONENTS (MODEL AND UTILITIES)
# ==============================================================================

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


# ==============================================================================
# 4. TRAINING FUNCTIONALITY
# ==============================================================================

TRAINER_CONFIG = {
    "stockfish_path": "/usr/games/stockfish",  # <-- IMPORTANT: Update this path for your system
    "model_path": "chessy_model.pth",
    "backup_model_path": "chessy_model_backup.pth",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "learning_rate": 1e-4,
    "num_games": 30,
    "num_epochs": 10,
    "stockfish_time_limit": 1.0,
    "search_depth": 1,
    "epsilon": 4
}

# --- Globals for worker processes ---
worker_model = None
worker_engine = None
worker_device = None
worker_config = None

def init_trainer_worker(model_state_dict, config):
    """Initializer for each worker process in the pool."""
    global worker_model, worker_engine, worker_device, worker_config
    
    worker_config = config
    worker_device = torch.device(config["device"])
    
    worker_model = NN1().to(worker_device)
    worker_model.load_state_dict(model_state_dict)
    worker_model.eval()
    
    try:
        worker_engine = eng.SimpleEngine.popen_uci(config["stockfish_path"])
    except FileNotFoundError:
        print(f"FATAL: Stockfish engine not found at {config['stockfish_path']}.")
        print("Please update the 'stockfish_path' in TRAINER_CONFIG.")
        raise

def _get_eval_worker(board):
    tensor = board_to_tensor(board).to(worker_device)
    with torch.no_grad():
        evaluation = worker_model(tensor)[0][0].item()
    return evaluation if board.turn == chess.WHITE else -evaluation

def _search_worker(board, depth, alpha, beta):
    if depth == 0 or board.is_game_over():
        return _get_eval_worker(board)

    max_eval = float('-inf')
    for move in board.legal_moves:
        board.push(move)
        eval_ = -_search_worker(board, depth - 1, -beta, -alpha)
        board.pop()
        max_eval = max(max_eval, eval_)
        alpha = max(alpha, eval_)
        if alpha >= beta:
            break
    return max_eval

def run_game_generation(engine_side):
    data = []
    move_count = 0
    board = chess.Board()
    lim = eng.Limit(time=worker_config["stockfish_time_limit"])

    while not board.is_game_over():
        is_bot_turn = (engine_side is None) or (board.turn != engine_side)
        
        if is_bot_turn:
            evaluations = {}
            for move in board.legal_moves:
                board.push(move)
                evaluations[move] = -_search_worker(board, depth=worker_config["search_depth"], alpha=float('-inf'), beta=float('inf'))
                board.pop()
            
            if not evaluations: break

            keys, logits = list(evaluations.keys()), torch.tensor(list(evaluations.values()))
            probs = torch.softmax(logits, dim=0)
            num_samples = min(worker_config["epsilon"], len(keys))
            best_indices = torch.multinomial(probs, num_samples=num_samples, replacement=False)
            final_move_idx = best_indices[torch.argmax(logits[best_indices])]
            move = keys[final_move_idx.item()]
        else:
            result = worker_engine.play(board, lim)
            move = result.move

        if is_bot_turn:
            data.append({'fen': board.fen(), 'move_number': move_count})

        board.push(move)
        move_count += 1

    result = board.result()
    score = {'1-0': 10.0, '0-1': -10.0}.get(result, 0.0)
    return data, score, move_count

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        print(f"Using device: {self.device}")
        self.model = NN1().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        self.criterion = nn.MSELoss()
        self._load_model()

    def _load_model(self):
        try:
            self.model.load_state_dict(torch.load(self.config["model_path"], map_location=self.device))
            print(f"Loaded model from {self.config['model_path']}")
        except FileNotFoundError:
            print("No model file found, starting from scratch.")

    def _train_on_batch(self, data, final_score, total_moves):
        if not data or total_moves == 0: return
        self.model.train()
        for entry in data:
            tensor = board_to_tensor(chess.Board(entry['fen'])).to(self.device)
            target_value = final_score * (1 - (entry['move_number'] / total_moves))
            target = torch.tensor(target_value, dtype=torch.float32).to(self.device)
            output = self.model(tensor)[0][0]
            loss = self.criterion(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print(f"Trained on a batch of {len(data)} positions. Final score: {final_score}")

    def run(self):
        num_procs = min(mp.cpu_count(), 8)
        mp.set_start_method('spawn', force=True)

        for i in range(self.config["num_epochs"]):
            print(f"\n--- Epoch {i+1}/{self.config['num_epochs']} ---")
            torch.save(self.model.state_dict(), self.config['backup_model_path'])
            
            num_games_per_type = self.config['num_games'] // 3
            game_params = ([chess.WHITE] * num_games_per_type + [chess.BLACK] * num_games_per_type + 
                           [None] * (self.config['num_games'] - 2 * num_games_per_type))
            
            try:
                init_args = (self.model.state_dict(), self.config)
                with mp.Pool(processes=num_procs, initializer=init_trainer_worker, initargs=init_args) as pool:
                    game_results = pool.map(run_game_generation, game_params)
            except Exception as e:
                print(f"A worker process failed: {e}. Training stopped.")
                break

            for data, score, mc in game_results:
                self._train_on_batch(data, score, mc)

            torch.save(self.model.state_dict(), self.config['model_path'])
            print(f"Epoch complete. Saved model to {self.config['model_path']}")
        
        print("\nTraining complete.")
        
        # Export to ONNX after training
        onnx_path = self.config['model_path'].replace('.pth', '.onnx')
        self.model.export_to_onnx(onnx_path, device='cpu')
        print(f"Model exported to ONNX: {onnx_path}")

# ==============================================================================
# 5. GRADIO WEB UI FUNCTIONALITY
# ==============================================================================

UI_CONFIG = {
    "local_model_path": "chessy_model.pth", # Path to the model trained by this script
    "search_depth": 2
}

class WebApp:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cpu")
        print(f"UI using device: {self.device}")
        self.model = NN1().to(self.device)
        self._load_model()
        self.model.eval()
        self.board = chess.Board()
        self.player_color = chess.WHITE
        self.current_eval = 0.0

    def _load_model(self):
        """Loads the locally trained model."""
        model_path = self.config["local_model_path"]
        try:
            if not os.path.exists(model_path):
                print(f"ERROR: Model file not found at '{model_path}'.")
                print("You must train the model first before launching the UI.")
                return
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Could not load the model. The AI will not function. Error: {e}")

    def _get_evaluation(self, board):
        tensor = board_to_tensor(board).to(self.device)
        with torch.no_grad():
            evaluation = self.model(tensor)[0][0].item()
        return evaluation if board.turn == chess.WHITE else -evaluation

    def _order_moves(self, board):
        return sorted(board.legal_moves, key=board.is_capture, reverse=True)

    def _search(self, board, depth, alpha, beta):
        if depth == 0 or board.is_game_over():
            return self._get_evaluation(board)
        max_eval = float('-inf')
        for move in self._order_moves(board):
            board.push(move)
            eval_ = -self._search(board, depth - 1, -beta, -alpha)
            board.pop()
            max_eval = max(max_eval, eval_)
            alpha = max(alpha, eval_)
            if alpha >= beta: break
        return max_eval
    
    def _ai_actor(self):
        best_move, max_eval = None, float('-inf')
        for move in self._order_moves(self.board):
            self.board.push(move)
            eval_ = -self._search(self.board, self.config['search_depth'] - 1, float('-inf'), float('inf'))
            self.board.pop()
            if eval_ > max_eval:
                max_eval, best_move = eval_, move
        self.current_eval = -max_eval
        return best_move

    def _board_to_image(self):
        if cairosvg is None: return np.zeros((400, 400, 3), dtype=np.uint8)
        svg_data = chess.svg.board(self.board, orientation=self.player_color)
        png_data = BytesIO()
        cairosvg.svg2png(bytestring=svg_data.encode('utf-8'), write_to=png_data, output_width=400, output_height=400)
        png_data.seek(0)
        return np.array(Image.open(png_data).convert("RGB"))

    def handle_move(self, move_uci):
        if self.board.is_game_over() or self.board.turn != self.player_color:
            status = "Game is over!" if self.board.is_game_over() else "It's not your turn!"
            return status, self._board_to_image(), f"Eval: {self.current_eval:.2f}"
        try:
            self.board.push_uci(move_uci)
        except ValueError:
            return f"Invalid move: {move_uci}", self._board_to_image(), f"Eval: {self.current_eval:.2f}"
        if self.board.is_game_over():
            return f"Game Over: {self.board.result()}", self._board_to_image(), f"Eval: {self.current_eval:.2f}"
        
        ai_move = self._ai_actor()
        if ai_move: self.board.push(ai_move)
        
        status = f"Game Over: {self.board.result()}" if self.board.is_game_over() else "Move accepted."
        return status, self._board_to_image(), f"Eval: {self.current_eval:.2f}"
    
    def handle_reset(self):
        self.board.reset()
        self.current_eval = 0.0
        return self._board_to_image(), "Board reset.", f"Eval: {self.current_eval:.2f}"

    def handle_set_color(self, color_choice):
        self.player_color = chess.WHITE if color_choice == "White" else chess.BLACK
        self.board.reset()
        self.current_eval = 0.0
        if self.player_color == chess.BLACK:
            ai_move = self._ai_actor()
            if ai_move: self.board.push(ai_move)
        return self._board_to_image()

    def launch(self):
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("## ♟️ NeoChess AI (Local Model)")
            with gr.Row():
                with gr.Column(scale=2):
                    board_output = gr.Image(value=self._board_to_image, label="Chess Board", type="numpy", interactive=False)
                with gr.Column(scale=1):
                    eval_val = gr.Markdown(f"Eval: {self.current_eval:.2f}")
                    color_choice = gr.Radio(choices=["White", "Black"], value="White", label="Choose Your Side")
                    reset_btn = gr.Button("♻️ Reset Board")
                    result_text = gr.Textbox(label="Status", interactive=False)
            
            move_input = gr.Textbox(label="Enter your move in UCI format (e.g., e2e4)", placeholder="e.g., e2e4")
            submit_btn = gr.Button("Play Move")
            
            submit_btn.click(self.handle_move, inputs=[move_input], outputs=[result_text, board_output, eval_val])
            reset_btn.click(self.handle_reset, inputs=[], outputs=[board_output, result_text, eval_val])
            color_choice.change(self.handle_set_color, inputs=[color_choice], outputs=[board_output])
        print("Launching Gradio UI...")
        demo.launch()


# ==============================================================================
# 6. EXECUTION INSTRUCTIONS FOR NOTEBOOKS
# ==============================================================================
#
# INSTRUCTIONS:
# 1. Run the entire script above this point in a single cell.
# 2. To TRAIN, uncomment and run the code below in a NEW cell.
# 3. To RUN THE UI, uncomment and run the code in a THIRD cell AFTER training.
#
# ------------------------------------------------------------------------------

# --- CELL 2: Training Code ---
print("Starting training process...")
trainer = Trainer(TRAINER_CONFIG)
trainer.run()

# ------------------------------------------------------------------------------

# --- CELL 3: UI Launch Code ---
print("Launching web application...")
webapp = WebApp(UI_CONFIG)
webapp.launch()
