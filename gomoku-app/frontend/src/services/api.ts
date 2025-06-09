import axios from 'axios';
import { GameState, GameConfig } from '../types/game';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const api = {
  // Create a new game
  createGame: async (playerName: string = 'Player', aiEnabled: boolean = true) => {
    const response = await axios.post(`${API_BASE_URL}/api/game/create`, {
      player_name: playerName,
      ai_enabled: aiEnabled,
    });
    // Map the snake_case response to camelCase
    return {
      gameId: response.data.game_id,
      playerName: response.data.player_name,
      aiEnabled: response.data.ai_enabled,
      boardSize: response.data.board_size,
    } as GameConfig;
  },

  // Get game state
  getGameState: async (gameId: string) => {
    const response = await axios.get<{ exists: boolean; board_state?: GameState }>(
      `${API_BASE_URL}/api/game/${gameId}/state`
    );
    return response.data;
  },

  // Make a move
  makeMove: async (gameId: string, action: number) => {
    const response = await axios.post<{
      success: boolean;
      board_state: GameState;
      ai_move?: number;
      message?: string;
    }>(`${API_BASE_URL}/api/game/${gameId}/move`, { action });
    return response.data;
  },

  // Get hint
  getHint: async (gameId: string) => {
    const response = await axios.post<{ suggested_move: number; score: number }>(
      `${API_BASE_URL}/api/game/${gameId}/hint`
    );
    return response.data;
  },

  // Delete game
  deleteGame: async (gameId: string) => {
    const response = await axios.delete(`${API_BASE_URL}/api/game/${gameId}`);
    return response.data;
  },
};

// WebSocket connection
export const createWebSocketConnection = (gameId: string) => {
  const wsUrl = `${API_BASE_URL.replace('http', 'ws')}/ws/${gameId}`;
  return new WebSocket(wsUrl);
};
