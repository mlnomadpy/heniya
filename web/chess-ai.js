/**
 * Chess AI Engine using ONNX.js and Negamax search
 */

class ChessAI {
    constructor() {
        this.session = null;
        this.isModelLoaded = false;
        this.pieceValues = {
            'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0,
            'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0
        };
        
        // Piece encoding for neural network input
        this.pieceEncoding = {
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
        };
    }

    async loadModel(modelPath) {
        try {
            console.log('Loading ONNX model from:', modelPath);
            this.session = new onnx.InferenceSession();
            await this.session.loadModel(modelPath);
            this.isModelLoaded = true;
            console.log('Model loaded successfully');
            return true;
        } catch (error) {
            console.error('Error loading model:', error);
            this.isModelLoaded = false;
            return false;
        }
    }

    async loadModelFromFile(file) {
        try {
            console.log('Loading ONNX model from file:', file.name);
            const arrayBuffer = await file.arrayBuffer();
            this.session = new onnx.InferenceSession();
            await this.session.loadModel(arrayBuffer);
            this.isModelLoaded = true;
            console.log('Model loaded successfully from file');
            return true;
        } catch (error) {
            console.error('Error loading model from file:', error);
            this.isModelLoaded = false;
            return false;
        }
    }

    boardToTensor(board) {
        /**
         * Convert chess board to tensor format expected by the neural network
         */
        const tensor = new Array(64).fill(0);
        const fen = board.fen().split(' ')[0]; // Get piece placement part of FEN
        
        let square = 0;
        for (let i = 0; i < fen.length; i++) {
            const char = fen[i];
            if (char === '/') {
                continue; // Skip rank separators
            } else if (/\d/.test(char)) {
                // Empty squares
                const numEmpty = parseInt(char);
                for (let j = 0; j < numEmpty; j++) {
                    tensor[square] = 0;
                    square++;
                }
            } else {
                // Piece
                tensor[square] = this.pieceEncoding[char] || 0;
                square++;
            }
        }
        
        return tensor;
    }

    async evaluatePosition(board) {
        /**
         * Evaluate position using the neural network
         */
        if (!this.isModelLoaded) {
            return this.simpleEvaluation(board);
        }

        try {
            const inputTensor = this.boardToTensor(board);
            const input = new onnx.Tensor(new Int32Array(inputTensor), 'int32', [1, 64]);
            
            const feeds = { 'board_input': input };
            const results = await this.session.run(feeds);
            
            const evaluation = results.evaluation.data[0];
            
            // Adjust evaluation based on whose turn it is
            return board.turn() === 'w' ? evaluation : -evaluation;
        } catch (error) {
            console.error('Error in neural network evaluation:', error);
            return this.simpleEvaluation(board);
        }
    }

    simpleEvaluation(board) {
        /**
         * Fallback evaluation function using material counting
         */
        if (board.in_checkmate()) {
            return board.turn() === 'w' ? -10000 : 10000;
        }
        
        if (board.in_stalemate() || board.in_threefold_repetition() || board.insufficient_material()) {
            return 0;
        }

        let score = 0;
        const squares = board.squares();
        
        for (let square in squares) {
            const piece = squares[square];
            if (piece) {
                const value = this.pieceValues[piece.type] || 0;
                score += piece.color === 'w' ? value : -value;
            }
        }
        
        // Add small random factor to avoid repetition
        score += (Math.random() - 0.5) * 0.1;
        
        return score;
    }

    orderMoves(board, moves) {
        /**
         * Order moves for better alpha-beta pruning
         */
        return moves.sort((a, b) => {
            const aCaptured = board.get(a.to);
            const bCaptured = board.get(b.to);
            
            // Prioritize captures of higher value pieces
            if (aCaptured && bCaptured) {
                return this.pieceValues[bCaptured.type] - this.pieceValues[aCaptured.type];
            } else if (aCaptured) {
                return -1;
            } else if (bCaptured) {
                return 1;
            }
            
            return 0;
        });
    }

    async negamax(board, depth, alpha, beta, color) {
        /**
         * Negamax search with alpha-beta pruning
         */
        if (depth === 0 || board.game_over()) {
            const eval_ = await this.evaluatePosition(board);
            return color * eval_;
        }

        let maxEval = -Infinity;
        const moves = this.orderMoves(board, board.moves({ verbose: true }));

        for (let move of moves) {
            board.move(move);
            const eval_ = -await this.negamax(board, depth - 1, -beta, -alpha, -color);
            board.undo();

            maxEval = Math.max(maxEval, eval_);
            alpha = Math.max(alpha, eval_);
            
            if (alpha >= beta) {
                break; // Alpha-beta cutoff
            }
        }

        return maxEval;
    }

    async getBestMove(board, depth = 2) {
        /**
         * Find the best move using negamax search
         */
        if (board.game_over()) {
            return null;
        }

        let bestMove = null;
        let bestEval = -Infinity;
        const color = board.turn() === 'w' ? 1 : -1;
        
        const moves = this.orderMoves(board, board.moves({ verbose: true }));
        
        for (let move of moves) {
            board.move(move);
            const eval_ = -await this.negamax(board, depth - 1, -Infinity, Infinity, -color);
            board.undo();

            if (eval_ > bestEval) {
                bestEval = eval_;
                bestMove = move;
            }
        }

        return {
            move: bestMove,
            evaluation: bestEval
        };
    }

    async getRandomMove(board) {
        /**
         * Get a random legal move (fallback)
         */
        const moves = board.moves({ verbose: true });
        if (moves.length === 0) return null;
        
        const randomIndex = Math.floor(Math.random() * moves.length);
        return {
            move: moves[randomIndex],
            evaluation: 0
        };
    }
}

// Export for use in other files
window.ChessAI = ChessAI;