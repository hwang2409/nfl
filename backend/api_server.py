"""
FastAPI Server for NFL Prediction API

This is a production-ready API server that can scale to thousands of users.
"""

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import redis
import json
import pickle
from pathlib import Path
import sys
from datetime import datetime
import os
from contextlib import asynccontextmanager
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from src.train_model_v2 import SimpleNFLModel
from src.predict_upcoming_v2 import predict_upcoming_v2
from src.database import (
    init_db, get_predictions, get_game_prediction, get_team_predictions,
    save_predictions, db_session, clear_database
)
from data_collection import get_current_season, load_game_data

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
MODEL_PATH = os.getenv("MODEL_PATH", "models/model_v2.pkl")

# Global state
redis_client = None
model_instance = None
model_version = None


def get_upcoming_weeks(season: int, num_weeks: int = 5) -> List[int]:
    """Get list of upcoming week numbers for the given season"""
    try:
        _, games = load_game_data()  # load_game_data returns (pbp, games) tuple
        if games is None or len(games) == 0:
            print("Warning: No game data available, using fallback week calculation")
            # Fallback: use current calendar week
            current_week = datetime.now().isocalendar()[1]
            return list(range(current_week, min(current_week + num_weeks, 19)))
        
        # Filter for current season
        season_games = games[games['season'] == season].copy()
        if len(season_games) == 0:
            print(f"Warning: No games found for season {season}")
            current_week = datetime.now().isocalendar()[1]
            return list(range(current_week, min(current_week + num_weeks, 19)))
        
        # Find current date
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Try to find date column
        date_col = None
        for col in ['game_date', 'gameday', 'date', 'game_time']:
            if col in season_games.columns:
                date_col = col
                break
        
        if date_col:
            # Get upcoming games
            upcoming_games = season_games[
                (season_games[date_col] >= current_date)
            ].copy()
            
            if len(upcoming_games) > 0:
                # Get unique weeks, sorted
                weeks = sorted(upcoming_games['week'].unique().tolist())[:num_weeks]
                return weeks
        
        # Fallback: use week column directly
        current_week = season_games['week'].min()
        max_week = season_games['week'].max()
        weeks = list(range(int(current_week), min(int(current_week) + num_weeks, int(max_week) + 1)))
        return weeks
        
    except Exception as e:
        print(f"Error getting upcoming weeks: {e}")
        # Fallback
        current_week = datetime.now().isocalendar()[1]
        return list(range(current_week, min(current_week + num_weeks, 19)))


def compute_predictions_for_week(season: int, week: int, model_version: str):
    """Compute and save predictions for a specific week"""
    global redis_client
    try:
        print(f"\n[Background] Computing predictions for {season} Week {week}...")
        predictions_df = predict_upcoming_v2(
            season=season,
            week=week,
            min_confidence=0.0
        )
        
        if predictions_df is not None and len(predictions_df) > 0:
            # Save to database
            try:
                save_predictions(predictions_df, model_version=model_version)
                print(f"✓ Saved {len(predictions_df)} predictions for Week {week}")
                
                # Cache results
                if redis_client:
                    cache_key = get_cache_key(season, week)
                    predictions = []
                    for _, row in predictions_df.iterrows():
                        predictions.append({
                            "season": int(row['season']),
                            "week": int(row['week']),
                            "home_team": row['home_team'],
                            "away_team": row['away_team'],
                            "home_win_probability": float(row['home_win_probability']),
                            "away_win_probability": float(row['away_win_probability']),
                            "predicted_winner": row['predicted_winner'],
                            "confidence": float(row['confidence']),
                            "game_date": str(row.get('gameday', '')) if pd.notna(row.get('gameday')) else None,
                            "gametime": str(row.get('gametime', '')) if pd.notna(row.get('gametime')) else None
                        })
                    set_cache(cache_key, predictions, ttl=3600, redis_client=redis_client)
            except Exception as e:
                print(f"✗ Error saving predictions for Week {week}: {e}")
        else:
            print(f"⚠️  No predictions generated for Week {week}")
    except Exception as e:
        print(f"✗ Error computing predictions for Week {week}: {e}")
        import traceback
        traceback.print_exc()


async def background_predictions_task(season: int, weeks: List[int], model_version: str):
    """Background task to compute predictions for multiple weeks"""
    for week in weeks:
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, compute_predictions_for_week, season, week, model_version)
        # Small delay between weeks
        await asyncio.sleep(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global redis_client, model_instance, model_version
    
    # Initialize Redis
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        redis_client.ping()
        print("✓ Redis connected")
    except Exception as e:
        print(f"⚠️  Redis connection failed: {e}")
        redis_client = None
    
    # Load model
    try:
        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model_instance = SimpleNFLModel.load(model_path)
        model_version = model_path.stat().st_mtime  # Use file mtime as version
        print(f"✓ Model loaded: {model_path}")
        print(f"  Threshold: {model_instance.threshold:.3f}")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        model_instance = None
    
    # Initialize database
    try:
        init_db(create_tables=True)
        print("✓ Database initialized")
    except Exception as e:
        print(f"⚠️  Database initialization failed: {e}")
        print("  API will continue but predictions won't be persisted")
    
    # Precompute predictions for next 2 weeks on startup
    if model_instance is not None:
        try:
            current_season = get_current_season()
            upcoming_weeks = get_upcoming_weeks(current_season, num_weeks=5)
            
            if len(upcoming_weeks) > 0:
                print(f"\n{'='*60}")
                print("Precomputing Predictions on Startup")
                print(f"{'='*60}")
                print(f"Season: {current_season}")
                print(f"Upcoming weeks: {upcoming_weeks}")
                
                # Synchronously compute predictions for first 2 weeks
                weeks_to_compute_now = upcoming_weeks[:2]
                weeks_to_compute_background = upcoming_weeks[2:5] if len(upcoming_weeks) > 2 else []
                
                print(f"\n[Immediate] Computing predictions for weeks: {weeks_to_compute_now}")
                for week in weeks_to_compute_now:
                    try:
                        print(f"\n[Startup] Computing predictions for {current_season} Week {week}...")
                        predictions_df = predict_upcoming_v2(
                            season=current_season,
                            week=week,
                            min_confidence=0.0
                        )
                        
                        if predictions_df is not None and len(predictions_df) > 0:
                            save_predictions(predictions_df, model_version=str(model_version))
                            print(f"✓ Saved {len(predictions_df)} predictions for Week {week}")
                            
                            # Cache results
                            if redis_client:
                                cache_key = get_cache_key(current_season, week)
                                predictions = []
                                for _, row in predictions_df.iterrows():
                                    predictions.append({
                                        "season": int(row['season']),
                                        "week": int(row['week']),
                                        "home_team": row['home_team'],
                                        "away_team": row['away_team'],
                                        "home_win_probability": float(row['home_win_probability']),
                                        "away_win_probability": float(row['away_win_probability']),
                                        "predicted_winner": row['predicted_winner'],
                                        "confidence": float(row['confidence']),
                                        "game_date": str(row.get('gameday', '')) if pd.notna(row.get('gameday')) else None,
                                        "gametime": str(row.get('gametime', '')) if pd.notna(row.get('gametime')) else None
                                    })
                                set_cache(cache_key, predictions, ttl=3600, redis_client=redis_client)
                        else:
                            print(f"⚠️  No predictions generated for Week {week}")
                    except Exception as e:
                        print(f"✗ Error computing predictions for Week {week}: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Start background task for remaining weeks (3-5)
                if weeks_to_compute_background:
                    print(f"\n[Background] Scheduling predictions for weeks: {weeks_to_compute_background}")
                    asyncio.create_task(background_predictions_task(
                        current_season, 
                        weeks_to_compute_background, 
                        str(model_version)
                    ))
                    print("✓ Background prediction tasks started")
                
                print(f"\n{'='*60}")
            else:
                print("⚠️  No upcoming weeks found")
        except Exception as e:
            print(f"⚠️  Error precomputing predictions: {e}")
            import traceback
            traceback.print_exc()
    
    yield
    
    # Cleanup
    if redis_client:
        redis_client.close()


# Create FastAPI app
app = FastAPI(
    title="NFL Prediction API",
    description="API for NFL game predictions using improved model",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class PredictionResponse(BaseModel):
    season: int
    week: int
    home_team: str
    away_team: str
    home_win_probability: float = Field(..., ge=0, le=1)
    away_win_probability: float = Field(..., ge=0, le=1)
    predicted_winner: str
    confidence: float = Field(..., ge=0, le=1)
    game_date: Optional[str] = None
    gametime: Optional[str] = None


class ModelStatus(BaseModel):
    loaded: bool
    version: Optional[float] = None
    threshold: Optional[float] = None
    cache_enabled: bool
    cache_connected: bool


# Dependency functions
def get_model():
    """Get model instance"""
    if model_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    return model_instance


def get_redis():
    """Get Redis client"""
    if redis_client is None:
        return None  # Cache disabled, continue without it
    return redis_client


# Helper functions
def get_cache_key(season: Optional[int], week: Optional[int], 
                  home_team: Optional[str] = None, 
                  away_team: Optional[str] = None) -> str:
    """Generate cache key"""
    if home_team and away_team:
        return f"prediction:{season}:{week}:{home_team}:{away_team}"
    elif season and week:
        return f"predictions:{season}:{week}"
    else:
        return f"predictions:upcoming"


def get_from_cache(key: str, redis_client: redis.Redis) -> Optional[dict]:
    """Get value from cache"""
    if redis_client is None:
        return None
    
    try:
        cached = redis_client.get(key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        print(f"Cache read error: {e}")
    return None


def set_cache(key: str, value: dict, ttl: int = 3600, redis_client: redis.Redis = None):
    """Set value in cache"""
    if redis_client is None:
        return
    
    try:
        redis_client.setex(key, ttl, json.dumps(value))
    except Exception as e:
        print(f"Cache write error: {e}")


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "NFL Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "predictions": "/api/v1/predictions/upcoming",
            "status": "/api/v1/status",
            "health": "/api/v1/health",
            "clear_database": "/api/v1/database/clear?confirm=true"
        }
    }


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_instance is not None,
        "cache_connected": redis_client is not None and redis_client.ping() if redis_client else False,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/status", response_model=ModelStatus)
async def model_status():
    """Get model status"""
    return ModelStatus(
        loaded=model_instance is not None,
        version=model_version,
        threshold=model_instance.threshold if model_instance else None,
        cache_enabled=redis_client is not None,
        cache_connected=redis_client.ping() if redis_client else False
    )


@app.get("/api/v1/predictions/upcoming", response_model=List[PredictionResponse])
async def get_upcoming_predictions(
    season: Optional[int] = Query(None, description="Season year (default: current)"),
    week: Optional[int] = Query(None, description="Week number (default: upcoming)"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    redis_client: redis.Redis = Depends(get_redis),
    model: SimpleNFLModel = Depends(get_model)
):
    """
    Get predictions for upcoming games.
    
    This endpoint:
    1. Checks Redis cache first (fastest)
    2. Checks database if cache miss
    3. Generates predictions if not in database
    4. Saves to database and caches results
    """
    # Check cache
    cache_key = get_cache_key(season, week)
    cached = get_from_cache(cache_key, redis_client)
    
    if cached:
        # Filter by confidence if needed
        if min_confidence > 0:
            filtered = [
                p for p in cached 
                if p.get('confidence', 0) >= min_confidence
            ]
            return filtered
        return cached
    
    # Check database
    try:
        db_predictions_df = get_predictions(
            season=season,
            week=week,
            min_confidence=min_confidence if min_confidence > 0 else None
        )
        
        if len(db_predictions_df) > 0:
            # Convert to response format
            predictions = []
            for _, row in db_predictions_df.iterrows():
                predictions.append({
                    "season": int(row['season']),
                    "week": int(row['week']),
                    "home_team": row['home_team'],
                    "away_team": row['away_team'],
                    "home_win_probability": float(row['home_win_probability']),
                    "away_win_probability": float(row['away_win_probability']),
                    "predicted_winner": row['predicted_winner'],
                    "confidence": float(row['confidence']),
                    "game_date": str(row.get('gameday', '')) if pd.notna(row.get('gameday')) else None,
                    "gametime": str(row.get('gametime', '')) if pd.notna(row.get('gametime')) else None
                })
            
            # Cache results
            set_cache(cache_key, predictions, ttl=3600, redis_client=redis_client)
            
            # Filter by confidence if needed
            if min_confidence > 0:
                filtered = [
                    p for p in predictions 
                    if p.get('confidence', 0) >= min_confidence
                ]
                return filtered
            
            return predictions
    except Exception as e:
        print(f"Database query failed: {e}")
        # Continue to generate predictions
    
    # Generate predictions (cache miss and database miss)
    try:
        predictions_df = predict_upcoming_v2(
            season=season,
            week=week,
            min_confidence=min_confidence
        )
        
        if predictions_df is None or len(predictions_df) == 0:
            raise HTTPException(
                status_code=404,
                detail="No predictions found for the specified criteria"
            )
        
        # Save to database
        try:
            save_predictions(predictions_df, model_version=str(model_version))
        except Exception as e:
            print(f"Warning: Could not save to database: {e}")
        
        # Convert to response format
        predictions = []
        for _, row in predictions_df.iterrows():
            predictions.append({
                "season": int(row['season']),
                "week": int(row['week']),
                "home_team": row['home_team'],
                "away_team": row['away_team'],
                "home_win_probability": float(row['home_win_probability']),
                "away_win_probability": float(row['away_win_probability']),
                "predicted_winner": row['predicted_winner'],
                "confidence": float(row['confidence']),
                "game_date": str(row.get('gameday', '')) if pd.notna(row.get('gameday')) else None,
                "gametime": str(row.get('gametime', '')) if pd.notna(row.get('gametime')) else None
            })
        
        # Cache results (TTL: 1 hour, or until game starts)
        set_cache(cache_key, predictions, ttl=3600, redis_client=redis_client)
        
        return predictions
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating predictions: {str(e)}"
        )


@app.get("/api/v1/predictions/game/{home_team}/{away_team}", response_model=PredictionResponse)
async def get_game_prediction(
    home_team: str,
    away_team: str,
    season: Optional[int] = Query(None, description="Season year"),
    week: Optional[int] = Query(None, description="Week number"),
    redis_client: redis.Redis = Depends(get_redis),
    model: SimpleNFLModel = Depends(get_model)
):
    """Get prediction for a specific game"""
    # Check cache
    cache_key = get_cache_key(season, week, home_team, away_team)
    cached = get_from_cache(cache_key, redis_client)
    
    if cached:
        return cached
    
    # Check database first
    if season is not None and week is not None:
        try:
            db_pred = get_game_prediction(season, week, home_team, away_team)
            if db_pred:
                # Cache and return
                set_cache(cache_key, db_pred, ttl=3600, redis_client=redis_client)
                return db_pred
        except Exception as e:
            print(f"Database query failed: {e}")
    
    # Get from upcoming predictions (will check cache/db/generate)
    predictions = await get_upcoming_predictions(
        season=season,
        week=week,
        min_confidence=0.0,
        redis_client=redis_client,
        model=model
    )
    
    # Find specific game
    for pred in predictions:
        if (pred['home_team'] == home_team and 
            pred['away_team'] == away_team):
            # Cache individual game prediction
            set_cache(cache_key, pred, ttl=3600, redis_client=redis_client)
            return pred
    
    raise HTTPException(
        status_code=404,
        detail=f"Prediction not found for {away_team} @ {home_team}"
    )


@app.get("/api/v1/predictions/team/{team}/upcoming", response_model=List[PredictionResponse])
async def get_team_upcoming_predictions(
    team: str,
    season: Optional[int] = Query(None, description="Season year"),
    redis_client: redis.Redis = Depends(get_redis),
    model: SimpleNFLModel = Depends(get_model)
):
    """Get all upcoming predictions for a specific team"""
    # Try database first
    try:
        db_predictions_df = get_team_predictions(team, season=season)
        if len(db_predictions_df) > 0:
            predictions = []
            for _, row in db_predictions_df.iterrows():
                predictions.append({
                    "season": int(row['season']),
                    "week": int(row['week']),
                    "home_team": row['home_team'],
                    "away_team": row['away_team'],
                    "home_win_probability": float(row['home_win_probability']),
                    "away_win_probability": float(row['away_win_probability']),
                    "predicted_winner": row['predicted_winner'],
                    "confidence": float(row['confidence']),
                    "game_date": str(row.get('gameday', '')) if pd.notna(row.get('gameday')) else None,
                    "gametime": str(row.get('gametime', '')) if pd.notna(row.get('gametime')) else None
                })
            return predictions
    except Exception as e:
        print(f"Database query failed: {e}")
    
    # Fallback to general predictions endpoint
    predictions = await get_upcoming_predictions(
        season=season,
        week=None,
        min_confidence=0.0,
        redis_client=redis_client,
        model=model
    )
    
    # Filter by team
    team_predictions = [
        p for p in predictions
        if p['home_team'] == team or p['away_team'] == team
    ]
    
    if not team_predictions:
        raise HTTPException(
            status_code=404,
            detail=f"No upcoming predictions found for team {team}"
        )
    
    return team_predictions


@app.post("/api/v1/cache/invalidate")
async def invalidate_cache(
    pattern: str = Query(..., description="Cache key pattern to invalidate"),
    redis_client: redis.Redis = Depends(get_redis)
):
    """Invalidate cache entries (admin endpoint)"""
    if redis_client is None:
        raise HTTPException(
            status_code=503,
            detail="Cache not available"
        )
    
    try:
        keys = redis_client.keys(pattern)
        if keys:
            redis_client.delete(*keys)
            return {
                "status": "success",
                "keys_deleted": len(keys),
                "pattern": pattern
            }
        return {
            "status": "success",
            "keys_deleted": 0,
            "pattern": pattern
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error invalidating cache: {str(e)}"
        )


@app.post("/api/v1/database/clear")
async def clear_db(
    confirm: bool = Query(False, description="Must be True to confirm database clearing")
):
    """
    Clear all data from the database (admin endpoint).
    
    WARNING: This will delete ALL predictions, games, and model versions.
    This action cannot be undone.
    
    Set confirm=True to proceed.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set confirm=True to clear database. This action cannot be undone."
        )
    
    try:
        result = clear_database()
        return {
            "status": "success",
            "message": "Database cleared successfully",
            "deleted": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing database: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Disable in production
        workers=1  # Increase in production with proper process manager
    )

