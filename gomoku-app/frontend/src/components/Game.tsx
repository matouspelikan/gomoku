import React, { useState, useEffect, useCallback } from 'react';
import GameBoard from './GameBoard';
import { GameState, GameConfig, WebSocketMessage } from '../types/game';
import { api, createWebSocketConnection } from '../services/api';
import './Game.css';

const Game: React.FC = () => {
  const [gameConfig, setGameConfig] = useState<GameConfig | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hint, setHint] = useState<number | undefined>();
  const [lastMove, setLastMove] = useState<number | undefined>();
  const [playerName, setPlayerName] = useState('Player');
  const [aiEnabled, setAiEnabled] = useState(true);
  const [showNewGameForm, setShowNewGameForm] = useState(true);

  // Clean up WebSocket on unmount
  useEffect(() => {
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [ws]);

  // Create a new game
  const createNewGame = async (name: string, useAI: boolean) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const config = await api.createGame(name, useAI);
      setGameConfig(config);
      setShowNewGameForm(false);
      
      // Get initial game state
      const stateResponse = await api.getGameState(config.gameId);
      if (stateResponse.exists && stateResponse.board_state) {
        setGameState(stateResponse.board_state);
      }
      
      // Set up WebSocket connection
      const websocket = createWebSocketConnection(config.gameId);
      
      websocket.onopen = () => {
        console.log('WebSocket connected');
      };
      
      websocket.onmessage = (event) => {
        const message: WebSocketMessage = JSON.parse(event.data);
        handleWebSocketMessage(message);
      };
      
      websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('Connection error. Please refresh the page.');
      };
      
      websocket.onclose = () => {
        console.log('WebSocket disconnected');
      };
      
      setWs(websocket);
    } catch (err) {
      setError('Failed to create game. Please try again.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle WebSocket messages
  const handleWebSocketMessage = (message: WebSocketMessage) => {
    switch (message.type) {
      case 'game_state':
        if (message.data) {
          setGameState(message.data);
        }
        break;
      
      case 'move_result':
        if (message.board_state) {
          setGameState(message.board_state);
          if (!message.success) {
            setError(message.message || 'Invalid move');
          }
        }
        break;
      
      case 'ai_move':
        if (message.board_state && message.action !== undefined) {
          setGameState(message.board_state);
          setLastMove(message.action);
          setTimeout(() => setLastMove(undefined), 2000);
        }
        break;
      
      case 'hint':
        if (message.suggested_move !== undefined) {
          setHint(message.suggested_move);
          setTimeout(() => setHint(undefined), 3000);
        }
        break;
      
      case 'error':
        setError(message.message || 'An error occurred');
        break;
    }
  };

  // Handle cell click
  const handleCellClick = useCallback((row: number, col: number) => {
    if (!gameConfig || !gameState || gameState.winner !== null) return;
    
    const action = row * gameState.board.length + col;
    
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'move', action }));
    } else {
      // Fallback to REST API
      makeMove(action);
    }
  }, [gameConfig, gameState, ws]);

  // Make move via REST API
  const makeMove = async (action: number) => {
    if (!gameConfig) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await api.makeMove(gameConfig.gameId, action);
      if (response.success) {
        setGameState(response.board_state);
        if (response.ai_move !== undefined) {
          setLastMove(response.ai_move);
          setTimeout(() => setLastMove(undefined), 2000);
        }
      } else {
        setError(response.message || 'Invalid move');
      }
    } catch (err) {
      setError('Failed to make move. Please try again.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  // Get hint
  const getHint = async () => {
    if (!gameConfig || !gameState || gameState.winner !== null) return;
    
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'get_hint' }));
    } else {
      // Fallback to REST API
      try {
        const response = await api.getHint(gameConfig.gameId);
        setHint(response.suggested_move);
        setTimeout(() => setHint(undefined), 3000);
      } catch (err) {
        setError('Failed to get hint');
        console.error(err);
      }
    }
  };

  // Reset game
  const resetGame = () => {
    if (ws) {
      ws.close();
    }
    setGameConfig(null);
    setGameState(null);
    setWs(null);
    setError(null);
    setHint(undefined);
    setLastMove(undefined);
    setShowNewGameForm(true);
  };

  // Get status message
  const getStatusMessage = () => {
    if (!gameState) return '';
    
    if (gameState.winner === 0) {
      return aiEnabled ? 'You win! 🎉' : 'Player 1 wins! 🎉';
    } else if (gameState.winner === 1) {
      return aiEnabled ? 'AI wins! 🤖' : 'Player 2 wins! 🎉';
    } else if (gameState.winner === 2) {
      return 'It\'s a draw! 🤝';
    } else {
      const currentPlayer = gameState.to_play === 0 ? 
        (aiEnabled ? 'Your' : 'Player 1\'s') : 
        (aiEnabled ? 'AI\'s' : 'Player 2\'s');
      return `${currentPlayer} turn`;
    }
  };

  if (showNewGameForm) {
    return (
      <div className="game-container">
        <div className="game-header">
          <h1>Gomoku (Piškvorky) 🎮</h1>
          <p>Connect 5 stones in a row to win!</p>
        </div>
        
        <div className="new-game-form">
          <h2>Start New Game</h2>
          <div className="form-group">
            <label htmlFor="playerName">Player Name:</label>
            <input
              id="playerName"
              type="text"
              value={playerName}
              onChange={(e) => setPlayerName(e.target.value)}
              placeholder="Enter your name"
            />
          </div>
          
          <div className="form-group">
            <label>
              <input
                type="checkbox"
                checked={aiEnabled}
                onChange={(e) => setAiEnabled(e.target.checked)}
              />
              Play against AI
            </label>
          </div>
          
          <button
            className="btn btn-primary"
            onClick={() => createNewGame(playerName, aiEnabled)}
            disabled={isLoading}
          >
            {isLoading ? 'Creating...' : 'Start Game'}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="game-container">
      <div className="game-header">
        <h1>Gomoku (Piškvorky) 🎮</h1>
        <div className="game-info">
          <span className="player-info">
            {aiEnabled ? `${playerName} vs AI` : 'Two Players'}
          </span>
          <span className="status">{getStatusMessage()}</span>
        </div>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {gameState?.winner !== null && (
        <div className="winner-overlay">
          <div className="winner-modal">
            <h2>{getStatusMessage()}</h2>
            <p>Game Over!</p>
            <button className="btn btn-primary" onClick={resetGame}>
              Play Again
            </button>
          </div>
        </div>
      )}

      <GameBoard
        gameState={gameState}
        onCellClick={handleCellClick}
        hint={hint}
        lastMove={lastMove}
        disabled={isLoading || gameState?.winner !== null}
      />

      <div className="game-controls">
        <button
          className="btn btn-secondary"
          onClick={getHint}
          disabled={isLoading || gameState?.winner !== null}
        >
          💡 Get Hint
        </button>
        
        <button
          className="btn btn-primary"
          onClick={resetGame}
        >
          🔄 New Game
        </button>
      </div>

      {gameState && (
        <div className="game-legend">
          <div className="legend-item">
            <div className="legend-stone player1-stone"></div>
            <span>{aiEnabled ? playerName : 'Player 1'} (Blue)</span>
          </div>
          <div className="legend-item">
            <div className="legend-stone player2-stone"></div>
            <span>{aiEnabled ? 'AI' : 'Player 2'} (Red)</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default Game;
