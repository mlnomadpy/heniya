# Heniya Chess AI

A chess AI application that uses neural networks for position evaluation and can be played through a web interface. The project includes:

- PyTorch-based neural network training for chess position evaluation
- ONNX model export for client-side inference
- Web application using ONNX.js for AI inference
- Negamax search algorithm combined with neural network evaluation

## Features

- **Neural Network Training**: Train chess position evaluation models using PyTorch
- **ONNX Export**: Export trained models to ONNX format for web deployment
- **Web Interface**: Play chess against the AI directly in your browser
- **Client-side AI**: No server required - AI runs entirely in the browser using ONNX.js
- **Configurable Search**: Adjustable search depth for AI difficulty
- **Move History**: Track and review game moves
- **Responsive Design**: Works on desktop and mobile devices

## Project Structure

```
heniya/
├── main.py              # Original training script (with ONNX export added)
├── nmn_replacement.py   # Fallback replacement for nmn package (for compatibility)
├── quick_train.py       # Quick training script to generate working models
├── test_model.py        # Test script for model creation and ONNX export
├── serve.py             # Simple HTTP server for web app
├── requirements.txt     # Python package dependencies
├── web/                 # Web application files
│   ├── index.html       # Main web page
│   ├── styles.css       # CSS styling
│   ├── chess-ai.js      # AI engine using ONNX.js and negamax
│   ├── app.js           # Main application logic
│   └── chessy_model.onnx # Trained chess model (generated)
├── chessy_model.pth     # PyTorch model (generated)
└── chessy_model.onnx    # ONNX model (generated)
```

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Clone or download the repository**

## Usage

### Option 1: Quick Start (Recommended)

1. **Generate a basic chess model:**
   ```bash
   python quick_train.py
   ```

2. **Start the web server:**
   ```bash
   python serve.py
   ```

3. **Open your browser and go to:**
   ```
   http://localhost:8000
   ```

4. **Play chess against the AI!**

### Option 2: Full Training

1. **Install Stockfish** (required for full training):
   - Ubuntu/Debian: `sudo apt-get install stockfish`
   - macOS: `brew install stockfish`
   - Windows: Download from [stockfish website](https://stockfishchess.org/download/)

2. **Update the Stockfish path in main.py** (line 104):
   ```python
   "stockfish_path": "/path/to/your/stockfish",
   ```

3. **Run the full training script:**
   ```python
   # In Python
   from main import Trainer, TRAINER_CONFIG
   trainer = Trainer(TRAINER_CONFIG)
   trainer.run()  # This will also export to ONNX
   ```

4. **Start the web server and play:**
   ```bash
   python serve.py
   ```

## How It Works

### Training
- Uses self-play games against Stockfish to generate training data
- Neural network learns to evaluate chess positions
- Combines attention mechanisms with custom neural modules
- Exports trained models to both PyTorch (.pth) and ONNX (.onnx) formats

### Web AI Engine
- Loads ONNX models in the browser using ONNX.js
- Implements negamax search with alpha-beta pruning
- Combines neural network position evaluation with traditional search
- Falls back to material-based evaluation if no model is loaded

### Architecture
- **Neural Network**: Embedding layer + Multi-head attention + Custom neural modules
- **Search Algorithm**: Negamax with alpha-beta pruning
- **Web Interface**: Chess.js for game logic + Chessboard.js for visualization
- **Model Format**: ONNX for cross-platform compatibility

## Web Application Features

- **Drag & Drop**: Intuitive piece movement
- **Player Color Choice**: Play as white or black
- **Difficulty Settings**: Adjustable AI search depth
- **Move History**: Complete game record
- **Model Loading**: Upload custom .onnx models
- **Responsive Design**: Works on all screen sizes

## Customization

### Training Your Own Model
1. Modify the neural network architecture in `NN1` class
2. Adjust training parameters in `TRAINER_CONFIG`
3. Run training with your preferred settings
4. The model will automatically be exported to ONNX format

### Custom Models
- Upload any compatible .onnx chess model through the web interface
- Models should accept 64-element integer arrays (chess board representation)
- Output should be a single evaluation score

## Technical Details

### Model Input Format
- 64-element integer array representing the chess board
- Piece encoding: Empty=0, P=1, N=2, B=3, R=4, Q=5, K=6, p=7, n=8, b=9, r=10, q=11, k=12
- Array represents squares a1-h1, a2-h2, ..., a8-h8

### Dependencies
- **Backend**: PyTorch, python-chess, ONNX, nmn, NumPy, Pillow, Gradio, CairoSVG
- **Frontend**: ONNX.js, Chess.js, Chessboard.js, jQuery

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Troubleshooting

### Model Loading Issues
- Ensure ONNX.js is loaded before the chess AI scripts
- Check browser console for error messages
- Verify model file format and compatibility

### Training Issues
- Make sure Stockfish is installed and path is correct
- Reduce training parameters if running out of memory
- Check that all dependencies are properly installed

### Web App Issues
- Ensure you're running the server (python serve.py)
- Check that all files are in the correct web/ directory
- Verify internet connection for CDN resources

## Future Improvements

- [ ] Better opening book integration
- [ ] Endgame tablebase support
- [ ] Tournament mode with multiple AIs
- [ ] Position analysis tools
- [ ] Cloud model sharing
- [ ] WebGL acceleration for inference