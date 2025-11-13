"""
Database Models and Connection for NFL Predictions

Provides database persistence for predictions with optimized indexes for hot lookups.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Index, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import JSONB
from contextlib import contextmanager

Base = declarative_base()


class Prediction(Base):
    """Prediction model for database storage"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    season = Column(Integer, nullable=False, index=True)
    week = Column(Integer, nullable=False, index=True)
    home_team = Column(String(3), nullable=False, index=True)
    away_team = Column(String(3), nullable=False, index=True)
    game_id = Column(String(50), nullable=True, index=True)  # Optional game identifier
    home_win_probability = Column(Float, nullable=False)
    away_win_probability = Column(Float, nullable=False)
    predicted_winner = Column(String(3), nullable=False)
    confidence = Column(Float, nullable=False, index=True)
    game_date = Column(DateTime, nullable=True, index=True)
    gametime = Column(String(20), nullable=True)
    model_version = Column(String(50), nullable=True)
    features = Column(JSONB, nullable=True)  # Store feature values for analysis
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Composite indexes for common query patterns
    __table_args__ = (
        # Hot lookup: Get predictions for a specific week
        Index('idx_season_week', 'season', 'week'),
        # Hot lookup: Get predictions for a team
        Index('idx_home_team_season', 'home_team', 'season'),
        Index('idx_away_team_season', 'away_team', 'season'),
        # Hot lookup: Get predictions by date range
        Index('idx_game_date', 'game_date'),
        # Hot lookup: Get high-confidence predictions
        Index('idx_confidence_season', 'confidence', 'season'),
        # Hot lookup: Get specific game
        Index('idx_teams_season_week', 'home_team', 'away_team', 'season', 'week'),
        # Hot lookup: Get recent predictions
        Index('idx_created_at', 'created_at'),
    )


class Game(Base):
    """Game model for tracking actual game results"""
    __tablename__ = 'games'
    
    game_id = Column(String(50), primary_key=True)
    season = Column(Integer, nullable=False, index=True)
    week = Column(Integer, nullable=False, index=True)
    home_team = Column(String(3), nullable=False, index=True)
    away_team = Column(String(3), nullable=False, index=True)
    game_date = Column(DateTime, nullable=True, index=True)
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
    status = Column(String(20), nullable=True, index=True)  # 'scheduled', 'in_progress', 'final'
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index('idx_games_season_week', 'season', 'week'),
        Index('idx_games_teams', 'home_team', 'away_team'),
    )


class ModelVersion(Base):
    """Model version tracking"""
    __tablename__ = 'model_versions'
    
    version_id = Column(String(50), primary_key=True)
    model_path = Column(String(255), nullable=False)
    training_date = Column(DateTime, nullable=False)
    accuracy = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    log_loss = Column(Float, nullable=True)
    features_count = Column(Integer, nullable=True)
    training_samples = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=False, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


# Database connection
_db_engine = None
_db_session_factory = None


def get_db_url() -> str:
    """Get database URL from environment variables"""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "nfl_predictions")
    user = os.getenv("POSTGRES_USER", "nfl_user")
    password = os.getenv("POSTGRES_PASSWORD", "nfl_password")
    
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def init_db(create_tables: bool = True):
    """Initialize database connection and create tables"""
    global _db_engine, _db_session_factory
    
    db_url = get_db_url()
    try:
        _db_engine = create_engine(
            db_url,
            pool_pre_ping=True,  # Verify connections before using
            pool_size=10,
            max_overflow=20,
            echo=False  # Set to True for SQL debugging
        )
        
        # Test connection
        with _db_engine.connect() as conn:
            from sqlalchemy import text
            conn.execute(text("SELECT 1"))
        
        _db_session_factory = sessionmaker(bind=_db_engine)
        
        if create_tables:
            Base.metadata.create_all(_db_engine)
        
        return _db_engine
    except Exception as e:
        error_msg = str(e)
        if "Connection refused" in error_msg or "could not connect" in error_msg.lower():
            raise ConnectionError(
                f"Could not connect to PostgreSQL at {db_url.split('@')[1]}. "
                f"Make sure PostgreSQL is running. "
                f"Start with: docker-compose up postgres"
            ) from e
        raise


def get_db_session() -> Session:
    """Get database session"""
    if _db_session_factory is None:
        init_db()
    return _db_session_factory()


@contextmanager
def db_session():
    """Context manager for database sessions"""
    session = get_db_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def save_predictions(predictions_df: pd.DataFrame, model_version: Optional[str] = None):
    """Save predictions DataFrame to database"""
    if predictions_df is None or len(predictions_df) == 0:
        return
    
    with db_session() as session:
        for _, row in predictions_df.iterrows():
            # Parse game date if available
            game_date = None
            if 'gameday' in row and pd.notna(row['gameday']):
                try:
                    game_date = pd.to_datetime(row['gameday'])
                except:
                    pass
            
            # Create or update prediction
            prediction = session.query(Prediction).filter(
                Prediction.season == int(row['season']),
                Prediction.week == int(row['week']),
                Prediction.home_team == row['home_team'],
                Prediction.away_team == row['away_team']
            ).first()
            
            if prediction is None:
                prediction = Prediction(
                    season=int(row['season']),
                    week=int(row['week']),
                    home_team=row['home_team'],
                    away_team=row['away_team'],
                    home_win_probability=float(row['home_win_probability']),
                    away_win_probability=float(row['away_win_probability']),
                    predicted_winner=row['predicted_winner'],
                    confidence=float(row['confidence']),
                    game_date=game_date,
                    gametime=str(row.get('gametime', '')) if pd.notna(row.get('gametime')) else None,
                    model_version=model_version
                )
                session.add(prediction)
            else:
                # Update existing prediction
                prediction.home_win_probability = float(row['home_win_probability'])
                prediction.away_win_probability = float(row['away_win_probability'])
                prediction.predicted_winner = row['predicted_winner']
                prediction.confidence = float(row['confidence'])
                prediction.game_date = game_date
                prediction.gametime = str(row.get('gametime', '')) if pd.notna(row.get('gametime')) else None
                prediction.model_version = model_version
                prediction.updated_at = datetime.utcnow()


def get_predictions(
    season: Optional[int] = None,
    week: Optional[int] = None,
    home_team: Optional[str] = None,
    away_team: Optional[str] = None,
    min_confidence: Optional[float] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """Get predictions from database"""
    with db_session() as session:
        query = session.query(Prediction)
        
        if season is not None:
            query = query.filter(Prediction.season == season)
        if week is not None:
            query = query.filter(Prediction.week == week)
        if home_team is not None:
            query = query.filter(Prediction.home_team == home_team)
        if away_team is not None:
            query = query.filter(Prediction.away_team == away_team)
        if min_confidence is not None:
            query = query.filter(Prediction.confidence >= min_confidence)
        
        query = query.order_by(Prediction.game_date, Prediction.gametime)
        
        if limit is not None:
            query = query.limit(limit)
        
        predictions = query.all()
        
        if not predictions:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for p in predictions:
            data.append({
                'season': p.season,
                'week': p.week,
                'home_team': p.home_team,
                'away_team': p.away_team,
                'home_win_probability': p.home_win_probability,
                'away_win_probability': p.away_win_probability,
                'predicted_winner': p.predicted_winner,
                'confidence': p.confidence,
                'gameday': p.game_date.strftime('%Y-%m-%d') if p.game_date else None,
                'gametime': p.gametime,
                'model_version': p.model_version,
                'created_at': p.created_at
            })
        
        return pd.DataFrame(data)


def get_game_prediction(
    season: int,
    week: int,
    home_team: str,
    away_team: str
) -> Optional[Dict]:
    """Get prediction for a specific game"""
    with db_session() as session:
        prediction = session.query(Prediction).filter(
            Prediction.season == season,
            Prediction.week == week,
            Prediction.home_team == home_team,
            Prediction.away_team == away_team
        ).first()
        
        if prediction is None:
            return None
        
        return {
            'season': prediction.season,
            'week': prediction.week,
            'home_team': prediction.home_team,
            'away_team': prediction.away_team,
            'home_win_probability': prediction.home_win_probability,
            'away_win_probability': prediction.away_win_probability,
            'predicted_winner': prediction.predicted_winner,
            'confidence': prediction.confidence,
            'game_date': prediction.game_date.strftime('%Y-%m-%d') if prediction.game_date else None,
            'gametime': prediction.gametime,
            'model_version': prediction.model_version
        }


def get_team_predictions(
    team: str,
    season: Optional[int] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """Get all predictions for a specific team"""
    with db_session() as session:
        query = session.query(Prediction).filter(
            (Prediction.home_team == team) | (Prediction.away_team == team)
        )
        
        if season is not None:
            query = query.filter(Prediction.season == season)
        
        query = query.order_by(Prediction.game_date, Prediction.gametime)
        
        if limit is not None:
            query = query.limit(limit)
        
        predictions = query.all()
        
        if not predictions:
            return pd.DataFrame()
        
        data = []
        for p in predictions:
            data.append({
                'season': p.season,
                'week': p.week,
                'home_team': p.home_team,
                'away_team': p.away_team,
                'home_win_probability': p.home_win_probability,
                'away_win_probability': p.away_win_probability,
                'predicted_winner': p.predicted_winner,
                'confidence': p.confidence,
                'gameday': p.game_date.strftime('%Y-%m-%d') if p.game_date else None,
                'gametime': p.gametime
            })
        
        return pd.DataFrame(data)


def delete_old_predictions(older_than_days: int = 365):
    """Delete predictions older than specified days"""
    cutoff_date = datetime.utcnow() - pd.Timedelta(days=older_than_days)
    
    with db_session() as session:
        deleted = session.query(Prediction).filter(
            Prediction.created_at < cutoff_date
        ).delete()
        
        return deleted

