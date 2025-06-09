import numpy as np
import random
from typing import Optional, List, Tuple


class GomokuGame:
    """Gomoku (Piškvorky) game implementation"""
    N = 15  # Board size
    ACTIONS = 225  # Total possible moves
    
    def __init__(self):
        self.board = np.zeros((self.N, self.N), dtype=np.int8)
        self.to_play = 0  # 0 for player 1, 1 for player 2
        self.winner = None
        self.move_history = []
    
    def valid_action(self, action: int) -> bool:
        """Check if an action is valid"""
        if self.winner is not None:
            return False
        row, col = action // self.N, action % self.N
        return 0 <= row < self.N and 0 <= col < self.N and self.board[row, col] == 0
    
    def valid_actions(self) -> List[int]:
        """Get list of valid actions"""
        if self.winner is not None:
            return []
        return [i for i in range(self.ACTIONS) if self.board[i // self.N, i % self.N] == 0]
    
    def make_move(self, action: int) -> bool:
        """Make a move on the board"""
        if not self.valid_action(action):
            return False
        
        row, col = action // self.N, action % self.N
        self.board[row, col] = self.to_play + 1
        self.move_history.append((int(row), int(col), int(self.to_play + 1)))
        
        # Check for winner
        self._check_winner(row, col)
        
        # Switch player
        self.to_play = 1 - self.to_play
        
        return True
    
    def _check_winner(self, last_row: int, last_col: int):
        """Check if the last move created a winning position"""
        player = self.board[last_row, last_col]
        
        # Directions: horizontal, vertical, diagonal1, diagonal2
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            
            # Check positive direction
            r, c = last_row + dr, last_col + dc
            while 0 <= r < self.N and 0 <= c < self.N and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
            
            # Check negative direction
            r, c = last_row - dr, last_col - dc
            while 0 <= r < self.N and 0 <= c < self.N and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            
            if count >= 5:
                self.winner = int(player - 1)
                return
        
        # Check for draw
        if len(self.valid_actions()) == 0:
            self.winner = 2  # Draw
    
    def get_board_state(self) -> dict:
        """Get the current board state as a dictionary"""
        return {
            "board": self.board.tolist(),
            "to_play": int(self.to_play),
            "winner": int(self.winner) if self.winner is not None else None,
            "move_history": self.move_history,
            "valid_actions": [int(x) for x in self.valid_actions()] if self.winner is None else []
        }
    
    def reset(self):
        """Reset the game to initial state"""
        self.board = np.zeros((self.N, self.N), dtype=np.int8)
        self.to_play = 0
        self.winner = None
        self.move_history = []


class GomokuHeuristic:
    """Heuristic AI player for Gomoku"""
    
    def __init__(self):
        self.coefs = [[0, 5, 25, 125, 1000], [0, 4, 20, 100, 400]]
        self.directions = [(-1, 1), (0, 1), (1, 1), (1, 0)]
    
    def get_move(self, game: GomokuGame) -> int:
        """Get the best move using heuristic evaluation"""
        board = game.board
        to_play = game.to_play
        
        # First move - play near center
        if np.all(board == 0):
            center = game.N // 2
            offset = random.randint(-1, 1)
            row = center + offset
            col = center + random.randint(-1, 1)
            return row * game.N + col
        
        best_score = -1
        best_moves = []
        
        for row in range(game.N):
            for col in range(game.N):
                if board[row, col] != 0:
                    continue
                
                score = self._evaluate_position(board, row, col, to_play)
                
                if score > best_score:
                    best_score = score
                    best_moves = [(row, col)]
                elif score == best_score:
                    best_moves.append((row, col))
        
        if best_moves:
            row, col = random.choice(best_moves)
            return int(row * game.N + col)
        
        return -1
    
    def _evaluate_position(self, board: np.ndarray, row: int, col: int, to_play: int) -> int:
        """Evaluate the score of placing a stone at the given position"""
        score = 0
        current_player = to_play + 1
        opponent = 2 - to_play
        
        for dr, dc in self.directions:
            for shift in range(-4, 1):
                counts = [0, 0, 0]  # [empty/out-of-bounds, current_player, opponent]
                
                for i in range(5):
                    r = row + dr * (shift + i)
                    c = col + dc * (shift + i)
                    
                    if r < 0 or r >= board.shape[0] or c < 0 or c >= board.shape[1]:
                        counts[0] += 1
                    else:
                        field = board[r, c]
                        if field == 0:
                            counts[0] += 1
                        elif field == current_player:
                            counts[1] += 1
                        else:
                            counts[2] += 1
                
                # If the line contains both players' stones, it's blocked
                if counts[1] > 0 and counts[2] > 0:
                    continue
                
                # If the line is not blocked, add score based on stone count
                if counts[1] > 0:
                    score += self.coefs[0][counts[1]]
                if counts[2] > 0:
                    score += self.coefs[1][counts[2]]
        
        return int(score)
