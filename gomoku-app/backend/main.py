from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List
import json
import uuid
from datetime import datetime

from game import GomokuGame, GomokuHeuristic

app = FastAPI(title="Gomoku Game API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active games in memory (in production, use Redis or database)
games: Dict[str, GomokuGame] = {}
ai_players: Dict[str, GomokuHeuristic] = {}


class CreateGameRequest(BaseModel):
    player_name: Optional[str] = "Player"
    ai_enabled: bool = True


class CreateGameResponse(BaseModel):
    game_id: str
    player_name: str
    ai_enabled: bool
    board_size: int


class MoveRequest(BaseModel):
    action: int


class MoveResponse(BaseModel):
    success: bool
    board_state: dict
    ai_move: Optional[int] = None
    message: Optional[str] = None


class GameStateResponse(BaseModel):
    exists: bool
    board_state: Optional[dict] = None


class HintRequest(BaseModel):
    pass


class HintResponse(BaseModel):
    suggested_move: int
    score: int


@app.get("/")
async def root():
    return {
        "message": "Gomoku Game API",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/game/create": "Create a new game",
            "GET /api/game/{game_id}/state": "Get game state",
            "POST /api/game/{game_id}/move": "Make a move",
            "POST /api/game/{game_id}/hint": "Get AI hint",
            "DELETE /api/game/{game_id}": "Delete a game",
            "WS /ws/{game_id}": "WebSocket connection for real-time updates"
        }
    }


@app.post("/api/game/create", response_model=CreateGameResponse)
async def create_game(request: CreateGameRequest):
    """Create a new game instance"""
    game_id = str(uuid.uuid4())
    games[game_id] = GomokuGame()
    
    if request.ai_enabled:
        ai_players[game_id] = GomokuHeuristic()
    
    return CreateGameResponse(
        game_id=game_id,
        player_name=request.player_name,
        ai_enabled=request.ai_enabled,
        board_size=GomokuGame.N
    )


@app.get("/api/game/{game_id}/state", response_model=GameStateResponse)
async def get_game_state(game_id: str):
    """Get the current state of a game"""
    if game_id not in games:
        return GameStateResponse(exists=False)
    
    game = games[game_id]
    return GameStateResponse(
        exists=True,
        board_state=game.get_board_state()
    )


@app.post("/api/game/{game_id}/move", response_model=MoveResponse)
async def make_move(game_id: str, request: MoveRequest):
    """Make a move in the game"""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = games[game_id]
    
    # Player move
    if not game.make_move(request.action):
        return MoveResponse(
            success=False,
            board_state=game.get_board_state(),
            message="Invalid move"
        )
    
    # If game is over after player move
    if game.winner is not None:
        return MoveResponse(
            success=True,
            board_state=game.get_board_state()
        )
    
    # AI move if enabled
    ai_move = None
    if game_id in ai_players and game.winner is None:
        ai = ai_players[game_id]
        ai_action = ai.get_move(game)
        if ai_action >= 0:
            game.make_move(ai_action)
            ai_move = int(ai_action)
    
    return MoveResponse(
        success=True,
        board_state=game.get_board_state(),
        ai_move=ai_move
    )


@app.post("/api/game/{game_id}/hint", response_model=HintResponse)
async def get_hint(game_id: str):
    """Get AI suggestion for the next move"""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = games[game_id]
    if game.winner is not None:
        raise HTTPException(status_code=400, detail="Game is already finished")
    
    # Use AI to get suggested move
    ai = GomokuHeuristic()
    suggested_action = ai.get_move(game)
    
    # Calculate score for the suggested move
    row, col = suggested_action // game.N, suggested_action % game.N
    score = ai._evaluate_position(game.board, row, col, game.to_play)
    
    return HintResponse(
        suggested_move=int(suggested_action),
        score=int(score)
    )


@app.delete("/api/game/{game_id}")
async def delete_game(game_id: str):
    """Delete a game instance"""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    del games[game_id]
    if game_id in ai_players:
        del ai_players[game_id]
    
    return {"message": "Game deleted successfully"}


# WebSocket endpoint for real-time game updates
@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    await websocket.accept()
    
    try:
        # Send initial game state
        if game_id in games:
            game = games[game_id]
            await websocket.send_json({
                "type": "game_state",
                "data": game.get_board_state()
            })
        else:
            await websocket.send_json({
                "type": "error",
                "message": "Game not found"
            })
            await websocket.close()
            return
        
        # Handle incoming messages
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "move":
                action = data["action"]
                game = games[game_id]
                
                # Player move
                if game.make_move(action):
                    await websocket.send_json({
                        "type": "move_result",
                        "success": True,
                        "board_state": game.get_board_state()
                    })
                    
                    # AI move if enabled and game not over
                    if game_id in ai_players and game.winner is None:
                        ai = ai_players[game_id]
                        ai_action = ai.get_move(game)
                        if ai_action >= 0:
                            game.make_move(ai_action)
                            await websocket.send_json({
                                "type": "ai_move",
                                "action": int(ai_action),
                                "board_state": game.get_board_state()
                            })
                else:
                    await websocket.send_json({
                        "type": "move_result",
                        "success": False,
                        "message": "Invalid move"
                    })
            
            elif data["type"] == "get_hint":
                game = games[game_id]
                if game.winner is None:
                    ai = GomokuHeuristic()
                    suggested_action = ai.get_move(game)
                    row, col = suggested_action // game.N, suggested_action % game.N
                    score = ai._evaluate_position(game.board, row, col, game.to_play)
                    
                    await websocket.send_json({
                        "type": "hint",
                        "suggested_move": int(suggested_action),
                        "score": int(score)
                    })
    
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for game {game_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
