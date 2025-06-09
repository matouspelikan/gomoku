import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import os
from pathlib import Path


class PisqorkyNet(nn.Module):
    """Neural network model for Piškvorky/Gomoku"""
    
    def __init__(self, board_size: int = 15):
        super().__init__()
        self.board_size = board_size
        self.input_channels = 3  # Empty, Player 1, Player 2
        
        # Convolutional layers
        layers: List[nn.Module] = []
        in_ch = self.input_channels
        for i in range(5):
            layers.append(nn.Conv2d(in_ch, 20 if i == 4 else 15, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_ch = 20 if i == 4 else 15
        self.stem = nn.Sequential(*layers)

        # Policy head
        self.policy_conv = nn.Conv2d(in_ch, 2, kernel_size=3, padding=1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # Value head
        self.value_conv = nn.Conv2d(in_ch, 2, kernel_size=3, padding=1)
        self.value_fc = nn.Linear(2 * board_size * board_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        
        # Policy head
        p = F.relu(self.policy_conv(x))
        p = self.policy_fc(p.flatten(1))
        p = F.softmax(p, dim=-1)
        
        # Value head
        v = F.relu(self.value_conv(x))
        v = torch.tanh(self.value_fc(v.flatten(1))).squeeze(-1)
        
        return p, v


class TorchGomokuAI:
    """PyTorch-based AI player for Gomoku"""
    
    def __init__(self, model_path: str, board_size: int = 15):
        self.board_size = board_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the model
        self.model = PisqorkyNet(board_size)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Using random weights.")
        
        self.model.to(self.device)
        self.model.eval()
    
    def _board_to_tensor(self, board: np.ndarray, to_play: int) -> torch.Tensor:
        """Convert game board to neural network input tensor"""
        # Create 3-channel representation: [empty, player1, player2]
        tensor = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        
        # Empty squares
        tensor[0] = (board == 0).astype(np.float32)
        
        # Player 1 stones
        tensor[1] = (board == 1).astype(np.float32)
        
        # Player 2 stones  
        tensor[2] = (board == 2).astype(np.float32)
        
        # If it's player 2's turn, swap the player channels
        if to_play == 1:
            tensor[1], tensor[2] = tensor[2].copy(), tensor[1].copy()
        
        # Add batch dimension and convert to torch tensor
        tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        return tensor
    
    def get_move(self, game) -> int:
        """Get the best move from the neural network"""
        from game import GomokuGame
        
        if len(game.valid_actions()) == 0:
            return -1
        
        # Convert board to tensor
        board_tensor = self._board_to_tensor(game.board, game.to_play)
        
        with torch.no_grad():
            policy, value = self.model(board_tensor)
            policy = policy.cpu().numpy()[0]  # Remove batch dimension
        
        # Mask invalid moves
        valid_actions = game.valid_actions()
        masked_policy = np.zeros_like(policy)
        for action in valid_actions:
            masked_policy[action] = policy[action]
        
        # Normalize the masked policy
        if np.sum(masked_policy) > 0:
            masked_policy = masked_policy / np.sum(masked_policy)
        else:
            # Fallback: uniform random among valid actions
            for action in valid_actions:
                masked_policy[action] = 1.0 / len(valid_actions)
        
        # Choose action with highest probability
        best_action = np.argmax(masked_policy)
        
        # Ensure the action is valid
        if best_action not in valid_actions:
            best_action = np.random.choice(valid_actions)
        
        return int(best_action)
    
    def get_move_with_policy(self, game) -> Tuple[int, np.ndarray]:
        """Get move and policy distribution for debugging/analysis"""
        if len(game.valid_actions()) == 0:
            return -1, np.zeros(self.board_size * self.board_size)
        
        board_tensor = self._board_to_tensor(game.board, game.to_play)
        
        with torch.no_grad():
            policy, value = self.model(board_tensor)
            policy = policy.cpu().numpy()[0]
        
        # Mask invalid moves
        valid_actions = game.valid_actions()
        masked_policy = np.zeros_like(policy)
        for action in valid_actions:
            masked_policy[action] = policy[action]
        
        if np.sum(masked_policy) > 0:
            masked_policy = masked_policy / np.sum(masked_policy)
        else:
            for action in valid_actions:
                masked_policy[action] = 1.0 / len(valid_actions)
        
        best_action = np.argmax(masked_policy)
        if best_action not in valid_actions:
            best_action = np.random.choice(valid_actions)
        
        return int(best_action), masked_policy


def create_torch_ai(model_path: str = None) -> TorchGomokuAI:
    """Factory function to create TorchGomokuAI instance"""
    if model_path is None:
        # Try to find the model in the current directory first
        current_dir = Path(__file__).parent
        model_path = current_dir / "model.pt"
        
        # If not found, try the parent directory
        if not model_path.exists():
            model_path = current_dir.parent / "pisqorky_cpp" / "model.pt"
        
    return TorchGomokuAI(str(model_path))
