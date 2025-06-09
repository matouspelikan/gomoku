import React from 'react';
import { GameState } from '../types/game';
import './GameBoard.css';

interface GameBoardProps {
  gameState: GameState | null;
  onCellClick: (row: number, col: number) => void;
  hint?: number;
  lastMove?: number;
  disabled?: boolean;
}

const GameBoard: React.FC<GameBoardProps> = ({
  gameState,
  onCellClick,
  hint,
  lastMove,
  disabled = false,
}) => {
  if (!gameState) {
    return <div className="game-board-loading">Loading game...</div>;
  }

  const boardSize = gameState.board.length;

  const getCellClass = (row: number, col: number, value: number): string => {
    const classes = ['cell'];
    const position = row * boardSize + col;
    
    if (value === 1) {
      classes.push('player1');
    } else if (value === 2) {
      classes.push('player2');
    }
    
    if (position === hint) {
      classes.push('hint');
    }
    
    if (position === lastMove) {
      classes.push('last-move');
    }
    
    if (disabled || gameState.winner !== null) {
      classes.push('disabled');
    }
    
    return classes.join(' ');
  };

  const isWinningCell = (row: number, col: number): boolean => {
    if (gameState.winner === null || gameState.winner === 2) return false;
    
    const directions = [
      [0, 1], [1, 0], [1, 1], [1, -1]
    ];
    
    const player = gameState.board[row][col];
    if (player === 0) return false;
    
    for (const [dr, dc] of directions) {
      let count = 1;
      
      // Check positive direction
      for (let i = 1; i < 5; i++) {
        const newRow = row + dr * i;
        const newCol = col + dc * i;
        if (
          newRow >= 0 && newRow < boardSize &&
          newCol >= 0 && newCol < boardSize &&
          gameState.board[newRow][newCol] === player
        ) {
          count++;
        } else {
          break;
        }
      }
      
      // Check negative direction
      for (let i = 1; i < 5; i++) {
        const newRow = row - dr * i;
        const newCol = col - dc * i;
        if (
          newRow >= 0 && newRow < boardSize &&
          newCol >= 0 && newCol < boardSize &&
          gameState.board[newRow][newCol] === player
        ) {
          count++;
        } else {
          break;
        }
      }
      
      if (count >= 5) return true;
    }
    
    return false;
  };

  return (
    <div className="game-board-container">
      <div className="game-board">
        {gameState.board.map((row, rowIndex) => (
          <div key={rowIndex} className="board-row">
            {row.map((cell, colIndex) => {
              const isWinner = isWinningCell(rowIndex, colIndex);
              return (
                <div
                  key={colIndex}
                  className={`${getCellClass(rowIndex, colIndex, cell)} ${
                    isWinner ? 'winning-cell' : ''
                  }`}
                  onClick={() => !disabled && cell === 0 && onCellClick(rowIndex, colIndex)}
                >
                  {cell === 1 && <div className="stone player1-stone" />}
                  {cell === 2 && <div className="stone player2-stone" />}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
};

export default GameBoard;
