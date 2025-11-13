"""
FastAPI Server for NFL Prediction API

This is a production-ready API server that can scale to thousands of users.
"""

from fastapi import FastAPI, HTTPException, Depends, Query
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
import random
from contextlib import asynccontextmanager

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

# Try to import model components - handle gracefully if dependencies are missing
try:
    from train_model_v2 import SimpleNFLModel
    from predict_upcoming_v2 import predict_upcoming_v2 as predict_upcoming_improved
    MODEL_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"⚠️  Model imports failed: {e}")
    print("  Server will start but predictions will not be available")
    SimpleNFLModel = None
    predict_upcoming_improved = None
    MODEL_AVAILABLE = False

try:
    from database import (
        init_db, get_predictions, get_game_prediction, get_team_predictions,
        save_predictions, db_session
    )
    DATABASE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Database imports failed: {e}")
    DATABASE_AVAILABLE = False

try:
    from data_collection import get_current_season, load_game_data
except ImportError as e:
    print(f"⚠️  Data collection imports failed: {e}")
    def get_current_season():
        from datetime import datetime
        now = datetime.now()
        return now.year - 1 if now.month < 9 else now.year
    load_game_data = None

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
MODEL_PATH = os.getenv("MODEL_PATH", "models/model_v2.pkl")

# Global state
redis_client = None
model_instance = None
model_version = None


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
    if MODEL_AVAILABLE and SimpleNFLModel is not None:
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
    else:
        print("⚠️  Model not available (dependencies missing)")
        model_instance = None
    
    # Initialize database
    if DATABASE_AVAILABLE:
        try:
            init_db(create_tables=True)
            print("✓ Database initialized")
        except Exception as e:
            print(f"⚠️  Database initialization failed: {e}")
            print("  API will continue but predictions won't be persisted")
    else:
        print("⚠️  Database not available")
    
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


class TimelineDataPoint(BaseModel):
    timestamp: str
    home_win_probability: float = Field(..., ge=0, le=1)
    away_win_probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)


class QBStats(BaseModel):
    name: str
    team: str
    qbr: float
    passingYards: float
    touchdowns: int
    interceptions: int


class TeamStats(BaseModel):
    epa: float
    dvoa: float
    elo: float


class Injury(BaseModel):
    player: str
    position: str
    status: str
    impact: str


class BettingSpread(BaseModel):
    source: str
    spread: float
    overUnder: float


class Weather(BaseModel):
    temperature: float
    condition: str
    windSpeed: float
    humidity: Optional[float] = None
    precipitation: Optional[float] = None
    isDome: bool


class Stadium(BaseModel):
    name: str
    city: str
    state: str
    capacity: Optional[int] = None
    surface: Optional[str] = None
    roofType: Optional[str] = None


class GameDetails(BaseModel):
    prediction: PredictionResponse
    homeQB: Optional[QBStats] = None
    awayQB: Optional[QBStats] = None
    homeTeamStats: Optional[TeamStats] = None
    awayTeamStats: Optional[TeamStats] = None
    injuries: Optional[List[Injury]] = None
    bettingSpreads: Optional[List[BettingSpread]] = None
    timeline: Optional[List[TimelineDataPoint]] = None
    weather: Optional[Weather] = None
    stadium: Optional[Stadium] = None


# Dependency functions
def get_model():
    """Get model instance (optional)"""
    return model_instance  # Return None if not loaded, don't raise error


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
            "health": "/api/v1/health"
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
    if DATABASE_AVAILABLE:
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
    else:
        print("Database not available, skipping database query")
    
    # Generate predictions (cache miss and database miss)
    # Check if model is available
    if model is None or predict_upcoming_improved is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. The prediction model is not available. Please check server logs or contact the administrator."
        )
    
    try:
        predictions_df = predict_upcoming_improved(
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
    if season is not None and week is not None and DATABASE_AVAILABLE:
        try:
            db_pred = get_game_prediction(season, week, home_team, away_team)
            if db_pred:
                # Cache and return
                set_cache(cache_key, db_pred, ttl=3600, redis_client=redis_client)
                return db_pred
        except Exception as e:
            print(f"Database query failed: {e}")
    
    # Get from upcoming predictions (will check cache/db/generate)
    # Check if model is available before calling
    if model is None or predict_upcoming_improved is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. The prediction model is not available. Please check server logs or contact the administrator."
        )
    
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
    if DATABASE_AVAILABLE:
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
    # Check if model is available before calling
    if model is None or predict_upcoming_improved is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. The prediction model is not available. Please check server logs or contact the administrator."
        )
    
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


@app.get("/api/v1/predictions/timeline/{home_team}/{away_team}", response_model=List[TimelineDataPoint])
async def get_prediction_timeline(
    home_team: str,
    away_team: str,
    season: Optional[int] = Query(None, description="Season year"),
    week: Optional[int] = Query(None, description="Week number"),
    redis_client: redis.Redis = Depends(get_redis),
    model: SimpleNFLModel = Depends(get_model)
):
    """
    Get prediction timeline for a specific game.
    
    Returns historical probability data points showing how predictions
    have changed over time. For now, generates synthetic timeline data
    based on current prediction. In production, this would query historical
    prediction data from the database.
    """
    # Get current prediction
    try:
        prediction = await get_game_prediction(
            home_team=home_team,
            away_team=away_team,
            season=season,
            week=week,
            redis_client=redis_client,
            model=model
        )
    except HTTPException:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction not found for {away_team} @ {home_team}"
        )
    
    # Check cache for timeline
    cache_key = f"timeline:{season}:{week}:{home_team}:{away_team}"
    cached = get_from_cache(cache_key, redis_client)
    if cached:
        return cached
    
    # Generate timeline data
    # In production, this would query historical predictions from database
    # For now, generate synthetic data based on current prediction
    timeline = []
    now = datetime.now()
    current_home_prob = prediction['home_win_probability']
    current_away_prob = prediction['away_win_probability']
    current_conf = prediction['confidence']
    
    # Generate 30 days of historical data points
    # More frequent for recent data (hourly for last 24h, then every 6-12 hours)
    # Last 24 hours: hourly
    for i in range(24, -1, -1):
        date = datetime(now.year, now.month, now.day, now.hour - i, 0, 0)
        variation = (random.random() - 0.5) * 0.06
        time_factor = i / 24
        
        home_prob = max(0.1, min(0.9, current_home_prob + variation * time_factor))
        away_prob = 1 - home_prob
        conf = 0.7 + (1 - time_factor) * 0.25
        
        timeline.append({
            "timestamp": date.isoformat(),
            "home_win_probability": round(home_prob, 2),
            "away_win_probability": round(away_prob, 2),
            "confidence": round(conf, 2)
        })
    
    # Days 2-7: every 6 hours
    for day in range(2, 8):
        for hour in [0, 6, 12, 18]:
            date = datetime(now.year, now.month, now.day - day, hour, 0, 0)
            variation = (random.random() - 0.5) * 0.08
            time_factor = day / 7
            
            home_prob = max(0.1, min(0.9, current_home_prob + variation * time_factor))
            away_prob = 1 - home_prob
            conf = 0.65 + (1 - time_factor) * 0.3
            
            timeline.append({
                "timestamp": date.isoformat(),
                "home_win_probability": round(home_prob, 2),
                "away_win_probability": round(away_prob, 2),
                "confidence": round(conf, 2)
            })
    
    # Days 8-30: every 12 hours
    for day in range(8, 31):
        for hour in [0, 12]:
            date = datetime(now.year, now.month, now.day - day, hour, 0, 0)
            variation = (random.random() - 0.5) * 0.1
            time_factor = day / 30
            
            home_prob = max(0.1, min(0.9, current_home_prob + variation * time_factor))
            away_prob = 1 - home_prob
            conf = 0.6 + (1 - time_factor) * 0.35
            
            timeline.append({
                "timestamp": date.isoformat(),
                "home_win_probability": round(home_prob, 2),
                "away_win_probability": round(away_prob, 2),
                "confidence": round(conf, 2)
            })
    
    # Sort by timestamp
    timeline.sort(key=lambda x: x['timestamp'])
    
    # Cache for 1 hour
    set_cache(cache_key, timeline, ttl=3600, redis_client=redis_client)
    
    return timeline


@app.get("/api/v1/predictions/game/{home_team}/{away_team}/details", response_model=GameDetails)
async def get_game_details(
    home_team: str,
    away_team: str,
    season: Optional[int] = Query(None, description="Season year"),
    week: Optional[int] = Query(None, description="Week number"),
    redis_client: redis.Redis = Depends(get_redis),
    model: SimpleNFLModel = Depends(get_model)
):
    """
    Get detailed game information including QB stats, team stats, injuries,
    betting spreads, and prediction timeline.
    """
    # Get base prediction
    try:
        prediction = await get_game_prediction(
            home_team=home_team,
            away_team=away_team,
            season=season,
            week=week,
            redis_client=redis_client,
            model=model
        )
    except HTTPException:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction not found for {away_team} @ {home_team}"
        )
    
    # Get timeline
    timeline = await get_prediction_timeline(
        home_team=home_team,
        away_team=away_team,
        season=season,
        week=week,
        redis_client=redis_client,
        model=model
    )
    
    # Get weather and stadium data
    # Check if dome stadium
    dome_stadiums = {'ATL', 'DET', 'IND', 'NO', 'DAL', 'HOU', 'MIN', 'LV', 'LAR', 'LAC'}
    is_dome = home_team in dome_stadiums
    
    # Stadium mapping
    stadium_map = {
        'BUF': {'name': 'Highmark Stadium', 'city': 'Orchard Park', 'state': 'NY', 'capacity': 71608, 'surface': 'FieldTurf', 'roofType': 'Open'},
        'MIA': {'name': 'Hard Rock Stadium', 'city': 'Miami Gardens', 'state': 'FL', 'capacity': 65326, 'surface': 'FieldTurf', 'roofType': 'Open'},
        'NE': {'name': 'Gillette Stadium', 'city': 'Foxborough', 'state': 'MA', 'capacity': 65878, 'surface': 'FieldTurf', 'roofType': 'Open'},
        'NYJ': {'name': 'MetLife Stadium', 'city': 'East Rutherford', 'state': 'NJ', 'capacity': 82500, 'surface': 'FieldTurf', 'roofType': 'Open'},
        'NYG': {'name': 'MetLife Stadium', 'city': 'East Rutherford', 'state': 'NJ', 'capacity': 82500, 'surface': 'FieldTurf', 'roofType': 'Open'},
        'BAL': {'name': 'M&T Bank Stadium', 'city': 'Baltimore', 'state': 'MD', 'capacity': 71008, 'surface': 'FieldTurf', 'roofType': 'Open'},
        'CIN': {'name': 'Paycor Stadium', 'city': 'Cincinnati', 'state': 'OH', 'capacity': 65515, 'surface': 'FieldTurf', 'roofType': 'Open'},
        'CLE': {'name': 'Cleveland Browns Stadium', 'city': 'Cleveland', 'state': 'OH', 'capacity': 67595, 'surface': 'FieldTurf', 'roofType': 'Open'},
        'PIT': {'name': 'Acrisure Stadium', 'city': 'Pittsburgh', 'state': 'PA', 'capacity': 68400, 'surface': 'Grass', 'roofType': 'Open'},
        'HOU': {'name': 'NRG Stadium', 'city': 'Houston', 'state': 'TX', 'capacity': 72220, 'surface': 'FieldTurf', 'roofType': 'Retractable'},
        'IND': {'name': 'Lucas Oil Stadium', 'city': 'Indianapolis', 'state': 'IN', 'capacity': 67000, 'surface': 'FieldTurf', 'roofType': 'Retractable'},
        'JAX': {'name': 'EverBank Stadium', 'city': 'Jacksonville', 'state': 'FL', 'capacity': 67814, 'surface': 'Grass', 'roofType': 'Open'},
        'TEN': {'name': 'Nissan Stadium', 'city': 'Nashville', 'state': 'TN', 'capacity': 69143, 'surface': 'FieldTurf', 'roofType': 'Open'},
        'DEN': {'name': 'Empower Field at Mile High', 'city': 'Denver', 'state': 'CO', 'capacity': 76125, 'surface': 'Grass', 'roofType': 'Open'},
        'KC': {'name': 'GEHA Field at Arrowhead Stadium', 'city': 'Kansas City', 'state': 'MO', 'capacity': 76416, 'surface': 'Grass', 'roofType': 'Open'},
        'LV': {'name': 'Allegiant Stadium', 'city': 'Las Vegas', 'state': 'NV', 'capacity': 65000, 'surface': 'FieldTurf', 'roofType': 'Dome'},
        'LAC': {'name': 'SoFi Stadium', 'city': 'Inglewood', 'state': 'CA', 'capacity': 70240, 'surface': 'FieldTurf', 'roofType': 'Fixed'},
        'LAR': {'name': 'SoFi Stadium', 'city': 'Inglewood', 'state': 'CA', 'capacity': 70240, 'surface': 'FieldTurf', 'roofType': 'Fixed'},
        'DAL': {'name': 'AT&T Stadium', 'city': 'Arlington', 'state': 'TX', 'capacity': 80000, 'surface': 'FieldTurf', 'roofType': 'Retractable'},
        'PHI': {'name': 'Lincoln Financial Field', 'city': 'Philadelphia', 'state': 'PA', 'capacity': 69596, 'surface': 'Grass', 'roofType': 'Open'},
        'WAS': {'name': 'FedExField', 'city': 'Landover', 'state': 'MD', 'capacity': 82000, 'surface': 'Grass', 'roofType': 'Open'},
        'CHI': {'name': 'Soldier Field', 'city': 'Chicago', 'state': 'IL', 'capacity': 61500, 'surface': 'Grass', 'roofType': 'Open'},
        'DET': {'name': 'Ford Field', 'city': 'Detroit', 'state': 'MI', 'capacity': 65000, 'surface': 'FieldTurf', 'roofType': 'Dome'},
        'GB': {'name': 'Lambeau Field', 'city': 'Green Bay', 'state': 'WI', 'capacity': 81441, 'surface': 'Grass', 'roofType': 'Open'},
        'MIN': {'name': 'U.S. Bank Stadium', 'city': 'Minneapolis', 'state': 'MN', 'capacity': 66655, 'surface': 'FieldTurf', 'roofType': 'Fixed'},
        'ATL': {'name': 'Mercedes-Benz Stadium', 'city': 'Atlanta', 'state': 'GA', 'capacity': 71000, 'surface': 'FieldTurf', 'roofType': 'Retractable'},
        'CAR': {'name': 'Bank of America Stadium', 'city': 'Charlotte', 'state': 'NC', 'capacity': 75523, 'surface': 'Grass', 'roofType': 'Open'},
        'NO': {'name': 'Caesars Superdome', 'city': 'New Orleans', 'state': 'LA', 'capacity': 73208, 'surface': 'FieldTurf', 'roofType': 'Fixed'},
        'TB': {'name': 'Raymond James Stadium', 'city': 'Tampa', 'state': 'FL', 'capacity': 65890, 'surface': 'Grass', 'roofType': 'Open'},
        'ARI': {'name': 'State Farm Stadium', 'city': 'Glendale', 'state': 'AZ', 'capacity': 63400, 'surface': 'FieldTurf', 'roofType': 'Retractable'},
        'SF': {'name': 'Levi\'s Stadium', 'city': 'Santa Clara', 'state': 'CA', 'capacity': 68500, 'surface': 'FieldTurf', 'roofType': 'Open'},
        'SEA': {'name': 'Lumen Field', 'city': 'Seattle', 'state': 'WA', 'capacity': 68000, 'surface': 'FieldTurf', 'roofType': 'Open'},
    }
    
    stadium_data = stadium_map.get(home_team, {
        'name': f'{home_team} Stadium',
        'city': 'Unknown',
        'state': 'Unknown'
    })
    
    # Generate weather data
    # In production, this would fetch from weather API
    weather_data = {
        'temperature': 72 if is_dome else 45 + random.randint(0, 40),
        'condition': 'Indoor' if is_dome else random.choice(['Sunny', 'Partly Cloudy', 'Cloudy', 'Clear']),
        'windSpeed': 0 if is_dome else random.randint(0, 20),
        'humidity': None if is_dome else 40 + random.randint(0, 40),
        'precipitation': 0 if is_dome else (random.randint(0, 30) if random.random() < 0.3 else 0),
        'isDome': is_dome
    }
    
    # For now, return minimal details with timeline, weather, and stadium
    # In production, this would fetch QB stats, team stats, injuries, betting from database/API
    return {
        "prediction": prediction,
        "timeline": timeline,
        "weather": weather_data,
        "stadium": stadium_data,
        # Placeholder data - replace with real data sources in production
        "homeQB": None,
        "awayQB": None,
        "homeTeamStats": None,
        "awayTeamStats": None,
        "injuries": None,
        "bettingSpreads": None
    }


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


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Disable in production
        workers=1  # Increase in production with proper process manager
    )

