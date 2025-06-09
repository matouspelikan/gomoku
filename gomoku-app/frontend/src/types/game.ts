export interface GameState {
  board: number[][];
  to_play: number;
  winner: number | null;
  move_history: Array<[number, number, number]>;
  valid_actions: number[];
}

export interface GameConfig {
  gameId: string;
  playerName: string;
  aiEnabled: boolean;
  boardSize: number;
}

export interface Move {
  action: number;
}

export interface WebSocketMessage {
  type: 'game_state' | 'move_result' | 'ai_move' | 'hint' | 'error';
  data?: GameState;
  board_state?: GameState;
  action?: number;
  suggested_move?: number;
  score?: number;
  success?: boolean;
  message?: string;
}
