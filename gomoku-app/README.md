# Gomoku (Piškvorky) Web Application

A full-stack web application for playing Gomoku (Five in a Row) with an AI opponent.

## Features

- 🎮 Play against AI or another player
- 🌐 Real-time gameplay using WebSockets
- 💡 Move hints to help you play better
- 🎨 Animated board with visual feedback
- 📱 Responsive design for mobile and desktop
- 🚀 Fast API backend with Python
- ⚛️ Modern React TypeScript frontend

## Tech Stack

- **Backend**: Python, FastAPI, WebSockets
- **Frontend**: React, TypeScript, WebSockets
- **Game Logic**: Custom heuristic AI based on position evaluation

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the backend server:
   ```bash
   python main.py
   ```

   The API will be available at `http://localhost:8000`

### Frontend Setup

1. In a new terminal, navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

   The app will open at `http://localhost:3000`

## How to Play

1. Enter your name and choose whether to play against AI
2. Click on any empty cell to place your stone
3. Connect 5 stones in a row (horizontal, vertical, or diagonal) to win
4. Use the "Get Hint" button for AI suggestions
5. Click "New Game" to start over

## API Endpoints

- `POST /api/game/create` - Create a new game
- `GET /api/game/{game_id}/state` - Get game state
- `POST /api/game/{game_id}/move` - Make a move
- `POST /api/game/{game_id}/hint` - Get AI hint
- `DELETE /api/game/{game_id}` - Delete a game
- `WS /ws/{game_id}` - WebSocket connection for real-time updates

## Game Rules

- Board size: 15x15
- Players take turns placing stones
- First player to connect 5 stones in a row wins
- Game ends in a draw if the board is full

## AI Heuristic

The AI uses a position evaluation heuristic that:
- Evaluates potential lines of 5 cells
- Assigns scores based on stone patterns
- Prioritizes both offensive and defensive moves
- Includes slight randomization for variety

## Future Enhancements (Azure Deployment Ready)

The application is designed with cloud deployment in mind:
- Containerizable with Docker
- Stateless design (can use Redis/PostgreSQL for persistence)
- Environment-based configuration
- Health check endpoints
- CORS configured for production

## Development

### Environment Variables

Create a `.env` file in the frontend directory:
```
REACT_APP_API_URL=http://localhost:8000
```

For production, update this to your Azure App Service URL.
