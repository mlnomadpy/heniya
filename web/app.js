/**
 * Main application for Heniya Chess AI
 */

class ChessApp {
    constructor() {
        this.game = new Chess();
        this.board = null;
        this.ai = new ChessAI();
        this.playerColor = 'white';
        this.aiDepth = 2;
        this.isPlayerTurn = true;
        this.moveHistory = [];
        this.gameInProgress = false;
        
        this.initializeBoard();
        this.setupEventListeners();
        this.updateUI();
        
        // Try to load default model
        this.tryLoadDefaultModel();
    }

    initializeBoard() {
        const config = {
            draggable: true,
            position: 'start',
            onDragStart: this.onDragStart.bind(this),
            onDrop: this.onDrop.bind(this),
            onSnapEnd: this.onSnapEnd.bind(this)
        };

        this.board = Chessboard('chessboard', config);
    }

    setupEventListeners() {
        // Player color selection
        document.getElementById('player-color').addEventListener('change', (e) => {
            this.playerColor = e.target.value;
            this.newGame();
        });

        // AI depth selection
        document.getElementById('ai-depth').addEventListener('change', (e) => {
            this.aiDepth = parseInt(e.target.value);
        });

        // Control buttons
        document.getElementById('new-game').addEventListener('click', () => {
            this.newGame();
        });

        document.getElementById('undo-move').addEventListener('click', () => {
            this.undoMove();
        });

        // Model loading
        document.getElementById('load-model-btn').addEventListener('click', () => {
            document.getElementById('model-upload').click();
        });

        document.getElementById('model-upload').addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.loadModelFromFile(file);
            }
        });
    }

    async tryLoadDefaultModel() {
        const modelPaths = [
            'chessy_model.onnx',
            '../chessy_model.onnx',
            'test_model.onnx',
            '../test_model.onnx'
        ];

        for (let path of modelPaths) {
            try {
                const success = await this.ai.loadModel(path);
                if (success) {
                    this.updateModelStatus('Model loaded successfully', 'loaded');
                    return;
                }
            } catch (error) {
                console.log(`Failed to load model from ${path}:`, error);
            }
        }

        this.updateModelStatus('No model found. Upload a .onnx model file or the AI will use basic evaluation.', 'error');
    }

    async loadModelFromFile(file) {
        this.updateModelStatus('Loading model...', '');
        try {
            const success = await this.ai.loadModelFromFile(file);
            if (success) {
                this.updateModelStatus(`Model loaded: ${file.name}`, 'loaded');
            } else {
                this.updateModelStatus('Failed to load model file', 'error');
            }
        } catch (error) {
            console.error('Error loading model:', error);
            this.updateModelStatus('Error loading model file', 'error');
        }
    }

    updateModelStatus(message, className) {
        const statusElement = document.getElementById('model-status');
        statusElement.textContent = message;
        statusElement.className = className;
    }

    onDragStart(source, piece, position, orientation) {
        // Don't pick up pieces if the game is over
        if (this.game.game_over()) return false;

        // Only pick up pieces for the side to move
        if ((this.game.turn() === 'w' && piece.search(/^b/) !== -1) ||
            (this.game.turn() === 'b' && piece.search(/^w/) !== -1)) {
            return false;
        }

        // Don't allow moves if it's not the player's turn
        if (!this.isPlayerTurn) return false;

        // Don't allow moves if the player color doesn't match the piece color
        if ((this.playerColor === 'white' && piece.search(/^b/) !== -1) ||
            (this.playerColor === 'black' && piece.search(/^w/) !== -1)) {
            return false;
        }
    }

    onDrop(source, target) {
        // See if the move is legal
        const move = this.game.move({
            from: source,
            to: target,
            promotion: 'q' // Always promote to a queen for simplicity
        });

        // Illegal move
        if (move === null) return 'snapback';

        this.moveHistory.push(move);
        this.updateMoveHistory();
        this.updateUI();

        // Make AI move after a short delay
        if (!this.game.game_over()) {
            this.isPlayerTurn = false;
            this.makeAIMove();
        }
    }

    onSnapEnd() {
        this.board.position(this.game.fen());
    }

    async makeAIMove() {
        this.showThinking(true);
        
        try {
            const result = await this.ai.getBestMove(this.game, this.aiDepth);
            
            if (result && result.move) {
                const move = this.game.move(result.move);
                if (move) {
                    this.moveHistory.push(move);
                    this.board.position(this.game.fen());
                    this.updateMoveHistory();
                    this.updateEvaluation(result.evaluation);
                }
            }
        } catch (error) {
            console.error('Error making AI move:', error);
            // Fallback to random move
            const randomResult = await this.ai.getRandomMove(this.game);
            if (randomResult && randomResult.move) {
                const move = this.game.move(randomResult.move);
                if (move) {
                    this.moveHistory.push(move);
                    this.board.position(this.game.fen());
                    this.updateMoveHistory();
                }
            }
        }

        this.showThinking(false);
        this.isPlayerTurn = true;
        this.updateUI();
    }

    newGame() {
        this.game.reset();
        this.board.start();
        this.moveHistory = [];
        this.updateMoveHistory();
        this.updateUI();
        this.gameInProgress = true;

        // If player chose black, make AI move first
        if (this.playerColor === 'black') {
            this.board.flip();
            this.isPlayerTurn = false;
            setTimeout(() => this.makeAIMove(), 500);
        } else {
            this.board.orientation('white');
            this.isPlayerTurn = true;
        }
    }

    undoMove() {
        if (this.moveHistory.length >= 2) {
            // Undo last two moves (player and AI)
            this.game.undo();
            this.game.undo();
            this.moveHistory.splice(-2);
            this.board.position(this.game.fen());
            this.updateMoveHistory();
            this.updateUI();
            this.isPlayerTurn = true;
        } else if (this.moveHistory.length === 1) {
            // Only undo player move
            this.game.undo();
            this.moveHistory.splice(-1);
            this.board.position(this.game.fen());
            this.updateMoveHistory();
            this.updateUI();
            this.isPlayerTurn = true;
        }
    }

    updateUI() {
        // Update game status
        let status = '';
        if (this.game.in_checkmate()) {
            status = 'Game over: ' + (this.game.turn() === 'w' ? 'Black' : 'White') + ' wins by checkmate!';
        } else if (this.game.in_draw()) {
            status = 'Game over: Draw';
        } else if (this.game.in_check()) {
            status = 'Check!';
        } else if (this.isPlayerTurn) {
            status = 'Your turn';
        } else {
            status = 'AI is thinking...';
        }

        document.getElementById('status').textContent = status;

        // Update undo button
        const undoButton = document.getElementById('undo-move');
        undoButton.disabled = this.moveHistory.length === 0 || !this.isPlayerTurn;

        this.gameInProgress = !this.game.game_over();
    }

    updateEvaluation(evaluation) {
        const evalElement = document.getElementById('evaluation');
        if (typeof evaluation === 'number') {
            const sign = evaluation >= 0 ? '+' : '';
            evalElement.textContent = `Evaluation: ${sign}${evaluation.toFixed(2)}`;
        }
    }

    showThinking(show) {
        const thinkingElement = document.getElementById('thinking');
        thinkingElement.style.display = show ? 'block' : 'none';
    }

    updateMoveHistory() {
        const moveListElement = document.getElementById('move-list');
        let html = '';

        for (let i = 0; i < this.moveHistory.length; i += 2) {
            const moveNumber = Math.floor(i / 2) + 1;
            const whiteMove = this.moveHistory[i] ? this.moveHistory[i].san : '';
            const blackMove = this.moveHistory[i + 1] ? this.moveHistory[i + 1].san : '';

            html += `<div class="move-pair">`;
            html += `<span class="move-number">${moveNumber}.</span>`;
            if (whiteMove) html += `<span class="move">${whiteMove}</span>`;
            if (blackMove) html += `<span class="move">${blackMove}</span>`;
            html += `</div>`;
        }

        moveListElement.innerHTML = html;
        
        // Scroll to bottom
        moveListElement.scrollTop = moveListElement.scrollHeight;
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.chessApp = new ChessApp();
});