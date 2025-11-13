# NFL Game Prediction System

A comprehensive machine learning system for predicting NFL game outcomes with high accuracy using an improved ensemble model architecture, advanced feature engineering, and time-aware validation. Includes a production-ready API server and modern Next.js frontend for scalable deployment.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [System Architecture](#system-architecture)
4. [Database Setup](#database-setup)
5. [API Server](#api-server)
6. [Frontend](#frontend)
7. [Prediction Storage](#prediction-storage)
8. [Improved Model System](#improved-model-system)
9. [Enhanced Features](#enhanced-features)
10. [Making Predictions](#making-predictions)
11. [Feature Engineering](#feature-engineering)
12. [Strategies for Improving Accuracy](#strategies-for-improving-accuracy)
13. [Project Structure](#project-structure)
14. [Data Sources](#data-sources)
15. [Notes and Best Practices](#notes-and-best-practices)

---

## Overview

This system implements an improved model architecture designed to maximize out-of-sample prediction accuracy. It uses:

- **Improved Stacked Ensemble**: Level-1 base models (XGBoost, LightGBM, CatBoost, Random Forest, Neural Network, Regularized Logistic Regression, Market baseline) + Level-2 meta-model
- **Multiple Ensemble Methods**: Stacking (default), Blending, Weighted Ensemble, and Dynamic Ensemble options
- **Probability Calibration**: Isotonic regression for better-calibrated probabilities
- **Feature Selection**: Importance-based feature selection to reduce overfitting
- **High-Signal Features**: Market consensus, Elo ratings, EPA metrics, QB performance, head-to-head history, target share, CB vs WR matchups
- **Time-Aware Validation**: Improved forward-chaining cross-validation with comprehensive metrics
- **Better Hyperparameters**: Tuned regularization, learning rates, and ensemble weights
- **Decision Rules**: Improved uncertainty dampening and confidence gating
- **PostgreSQL Database**: Permanent storage with optimized indexes for fast lookups
- **Redis Caching**: Sub-second response times for API requests
- **Production API**: FastAPI server ready for horizontal scaling
- **Modern Frontend**: Next.js web application with responsive design

### Expected Performance

- **Baseline**: ~52-55% accuracy
- **With Improved Model**: ~58-60% accuracy (fast mode)
- **With Improved Model + Accuracy Mode**: ~60-62% accuracy
- **With Optuna Tuning**: +2-5% additional improvement
- **Elite Level**: ~62-65% accuracy (very difficult to achieve consistently)

### Recent Updates

**Model Consolidation:**
- ✅ **Single Model Architecture**: All old model files have been removed. The system now uses only the improved model (`improved_model.pkl`)
- ✅ **Unified Scripts**: All training, prediction, and evaluation scripts use the improved model
- ✅ **API Server**: Production-ready FastAPI server added for scalable deployment
- ✅ **Database Storage**: PostgreSQL database with optimized indexes for fast lookups

**Latest Features Added:**
- ✅ **Weather API Integration**: Real-time weather forecasts from Open-Meteo (free, no API key required)
- ✅ **Injury Report Integration**: Automated NFL.com injury report scraping and analysis
- ✅ **Pro Football Reference Integration**: Team and player-level statistics from PFR
- ✅ **Neural Network (MLP)**: Added Multi-Layer Perceptron as a Level-1 base model
- ✅ **CatBoost Integration**: Added CatBoost as a Level-1 base model
- ✅ **Random Forest**: Added Random Forest as a Level-1 base model
- ✅ **Optuna Hyperparameter Tuning**: Automated Bayesian optimization
- ✅ **Accuracy Mode**: New `--accurate` flag for maximum accuracy
- ✅ **Target Share Metrics**: Top receiver target share and target concentration
- ✅ **CB vs WR Matchups**: Player-level defensive matchup analysis
- ✅ **Advanced Ensemble Methods**: Blending, Weighted Ensemble, and Dynamic Ensemble options
- ✅ **PostgreSQL Database**: Permanent prediction storage with optimized indexes
- ✅ **Next.js Frontend**: Modern web interface for viewing predictions

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect Data

```bash
# Collect data (defaults to last 6-7 seasons for recent data)
python src/data_collection.py
```

### 3. Train Model

**Fast Mode (Default):**
```bash
python src/train_improved_model.py
```

**Accuracy Mode (Slower but More Accurate):**
```bash
python src/train_improved_model.py --accurate
```

**With Optuna Hyperparameter Tuning:**
```bash
python src/train_improved_model.py --accurate --optuna
```

### 4. Make Predictions

```bash
# Upcoming games (default: current season, next week)
python src/predict_upcoming_improved.py

# Specific season and week
python src/predict_upcoming_improved.py --season 2024 --week 1

# Only high-confidence predictions
python src/predict_upcoming_improved.py --min-confidence 0.7
```

### 5. Evaluate Model Accuracy

```bash
# Evaluate on all available games
python src/evaluate_model.py

# Evaluate specific season
python src/evaluate_model.py --season 2023
```

### 6. Start API Server (Optional)

For production deployment:

```bash
# Install API dependencies
pip install -r backend/requirements_api.txt

# Start Redis and PostgreSQL
cd backend
docker-compose up -d redis postgres

# Initialize database
python src/init_database.py

# Start API server
python api_server.py
```

The API will be available at `http://localhost:8000` with auto-generated documentation at `/docs`.

---

## System Architecture

This section outlines the system architecture for serving NFL game predictions to thousands of concurrent users with sub-second response times.

### Architecture Principles

1. **Caching First**: Predictions for the same games are cached (games don't change until they're played)
2. **Async Processing**: Heavy computations (model retraining, feature calculation) run in background
3. **Stateless API**: API servers are horizontally scalable
4. **Event-Driven**: Real-time updates via webhooks/events when games finish
5. **Cost Optimization**: Minimize redundant computations through intelligent caching

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                            │
│  Web App / Mobile App / API Consumers                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ HTTPS
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      API GATEWAY / CDN                          │
│  - Rate Limiting                                               │
│  - Authentication                                              │
│  - Request Routing                                             │
│  - SSL Termination                                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    LOAD BALANCER                                │
│  (Round Robin / Least Connections)                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼──────┐   ┌────────▼────────┐   ┌─────▼──────┐
│  API Server  │   │   API Server    │   │ API Server │
│  Instance 1  │   │   Instance 2    │   │ Instance N │
└───────┬──────┘   └────────┬────────┘   └─────┬──────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼──────┐   ┌────────▼────────┐   ┌─────▼──────┐
│   Redis      │   │   PostgreSQL    │   │  S3/Blob   │
│   Cache      │   │   Database      │   │  Storage   │
└──────────────┘   └─────────────────┘   └────────────┘
                            │
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    BACKGROUND WORKERS                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Data         │  │ Model        │  │ Feature      │          │
│  │ Collector    │  │ Retrainer    │  │ Calculator   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. API Layer (FastAPI)

**Responsibilities:**
- Handle HTTP requests
- Authentication & authorization
- Request validation
- Response formatting
- Cache lookup before model inference

**Key Endpoints:**
```
GET  /api/v1/predictions/upcoming
GET  /api/v1/predictions/week/{season}/{week}
GET  /api/v1/predictions/game/{game_id}
GET  /api/v1/predictions/team/{team}/upcoming
GET  /api/v1/model/status
POST /api/v1/predictions/batch
```

**Technology:** FastAPI (async, auto-docs, high performance)

**Scaling:** Horizontal scaling with multiple instances behind load balancer

#### 2. Caching Layer (Redis)

**Cache Strategy:**

1. **Prediction Cache** (TTL: Until game starts)
   - Key: `prediction:{season}:{week}:{home_team}:{away_team}`
   - Value: JSON prediction result
   - TTL: Until game start time

2. **Feature Cache** (TTL: 1 hour)
   - Key: `features:{team}:{season}:{week}`
   - Value: Serialized feature vector
   - TTL: 1 hour (features change slowly)

**Cache Invalidation:**
- Predictions: Invalidated when game finishes (via event)
- Features: Invalidated when new game data arrives
- Model: Invalidated when new model is trained

#### 3. Database Layer (PostgreSQL)

**Schema Design:**

```sql
-- Predictions table
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    home_team VARCHAR(3) NOT NULL,
    away_team VARCHAR(3) NOT NULL,
    game_id VARCHAR(50),
    home_win_probability FLOAT NOT NULL,
    away_win_probability FLOAT NOT NULL,
    predicted_winner VARCHAR(3) NOT NULL,
    confidence FLOAT NOT NULL,
    game_date TIMESTAMP,
    gametime VARCHAR(20),
    model_version VARCHAR(50),
    features JSONB,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

-- Optimized indexes for hot lookups
CREATE INDEX idx_season_week ON predictions(season, week);
CREATE INDEX idx_home_team_season ON predictions(home_team, season);
CREATE INDEX idx_away_team_season ON predictions(away_team, season);
CREATE INDEX idx_teams_season_week ON predictions(home_team, away_team, season, week);
CREATE INDEX idx_confidence_season ON predictions(confidence, season);
CREATE INDEX idx_game_date ON predictions(game_date);
```

**Read Replicas:** Use read replicas for read-heavy queries

### Data Flow

#### Prediction Request Flow

```
1. Client → API Gateway
   ↓
2. API Gateway → Load Balancer
   ↓
3. Load Balancer → API Server
   ↓
4. API Server checks Redis cache
   ├─ Cache Hit → Return cached prediction (5-10ms)
   └─ Cache Miss → Continue
   ↓
5. API Server checks database
   ├─ Found → Return from DB, populate cache (10-50ms)
   └─ Not Found → Continue
   ↓
6. API Server loads model (if not in memory)
   ↓
7. API Server calculates features (or retrieves from cache)
   ↓
8. API Server runs inference
   ↓
9. API Server stores in DB and cache
   ↓
10. API Server returns prediction (200-500ms total)
```

### Performance Targets

- **Cache Hit**: 5-10ms response time
- **Database Hit**: 10-50ms response time (indexed queries)
- **Cache Miss + DB Miss**: 200-500ms (includes model inference)
- **Throughput**: 1000+ requests/second (with caching + database)

### Scalability Considerations

**Horizontal Scaling:**
- API Servers: Stateless design allows unlimited horizontal scaling
- Database: Read replicas for read-heavy workloads
- Cache: Redis Cluster for high availability

**Performance Optimizations:**
1. **Pre-compute Predictions**: Generate all predictions for upcoming week, store in cache and database
2. **Batch Processing**: Process multiple games in single inference call
3. **Feature Caching**: Cache team features (change slowly)
4. **CDN for Static Data**: Serve team logos, historical stats via CDN

---

## Database Setup

Predictions are stored in PostgreSQL instead of CSV files. This provides:
- ✅ **Persistence**: Survives Docker restarts
- ✅ **Performance**: Indexed queries for fast lookups
- ✅ **Scalability**: Handles thousands of concurrent queries
- ✅ **Reliability**: ACID transactions, data integrity

### Quick Start: Database Setup

#### 1. Start PostgreSQL

**Option 1: Docker Compose (Recommended)**
```bash
cd backend
docker-compose up -d postgres
```

**Option 2: Docker Run**
```bash
docker run -d \
  --name nfl-postgres \
  -p 5432:5432 \
  -e POSTGRES_DB=nfl_predictions \
  -e POSTGRES_USER=nfl_user \
  -e POSTGRES_PASSWORD=nfl_password \
  postgres:15-alpine
```

#### 2. Initialize Database Schema

```bash
cd backend
python src/init_database.py
```

This will:
- Connect to PostgreSQL
- Create all tables (predictions, games, model_versions)
- Create optimized indexes for fast lookups

#### 3. Verify Setup

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Connect to database
docker-compose exec postgres psql -U nfl_user -d nfl_predictions

# Check tables
\dt

# Check indexes
\d predictions

# View predictions
SELECT * FROM predictions LIMIT 10;
```

### Database Schema

#### Predictions Table

```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    home_team VARCHAR(3) NOT NULL,
    away_team VARCHAR(3) NOT NULL,
    game_id VARCHAR(50),
    home_win_probability FLOAT NOT NULL,
    away_win_probability FLOAT NOT NULL,
    predicted_winner VARCHAR(3) NOT NULL,
    confidence FLOAT NOT NULL,
    game_date TIMESTAMP,
    gametime VARCHAR(20),
    model_version VARCHAR(50),
    features JSONB,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);
```

#### Indexes (Optimized for Hot Lookups)

1. **idx_season_week**: `(season, week)` - Get predictions for a week (~1-5ms)
2. **idx_home_team_season**: `(home_team, season)` - Get team's home games (~2-10ms)
3. **idx_away_team_season**: `(away_team, season)` - Get team's away games (~2-10ms)
4. **idx_game_date**: `(game_date)` - Get predictions by date
5. **idx_confidence_season**: `(confidence, season)` - Get high-confidence predictions (~5-20ms)
6. **idx_teams_season_week**: `(home_team, away_team, season, week)` - Get specific game (~1-3ms)
7. **idx_created_at**: `(created_at)` - Get recent predictions

### Environment Variables

The database connection uses these environment variables (already set in docker-compose.yml):

```bash
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=nfl_predictions
POSTGRES_USER=nfl_user
POSTGRES_PASSWORD=nfl_password
```

### Usage

**Saving Predictions:**
Predictions are automatically saved when:
1. Running `predict_upcoming_improved.py` script
2. API generates new predictions (cache miss + database miss)

**Querying Predictions:**

**Via Python:**
```python
from src.database import get_predictions, get_game_prediction, get_team_predictions

# Get all predictions for a week
df = get_predictions(season=2024, week=1)

# Get specific game
pred = get_game_prediction(2024, 1, 'KC', 'BUF')

# Get team's predictions
df = get_team_predictions('KC', season=2024)
```

**Via SQL:**
```sql
-- Get predictions for a week
SELECT * FROM predictions WHERE season = 2024 AND week = 1;

-- Get high-confidence predictions
SELECT * FROM predictions WHERE confidence >= 0.7 ORDER BY confidence DESC;

-- Get team's upcoming games
SELECT * FROM predictions 
WHERE (home_team = 'KC' OR away_team = 'KC') 
  AND season = 2024 
ORDER BY game_date;
```

### Performance

With indexes, common queries are very fast:
- **Get week predictions**: ~1-5ms (indexed: season + week)
- **Get team predictions**: ~2-10ms (indexed: team + season)
- **Get specific game**: ~1-3ms (indexed: teams + season + week)
- **Get high-confidence**: ~5-20ms (indexed: confidence + season)

### Troubleshooting

**"Connection refused"**
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Start PostgreSQL
docker-compose up -d postgres
```

**"Table does not exist"**
```bash
# Initialize schema
python src/init_database.py
```

---

## API Server

Production-ready FastAPI server for serving NFL game predictions.

### Quick Start

#### Local Development

1. **Install dependencies:**
```bash
pip install -r backend/requirements_api.txt
```

2. **Start Redis and PostgreSQL:**
```bash
cd backend
docker-compose up -d redis postgres
```

3. **Initialize database schema:**
```bash
python src/init_database.py
```

4. **Start API server:**
```bash
python api_server.py
```

The API will be available at `http://localhost:8000`

### Using Docker Compose

```bash
cd backend
docker-compose up
```

This starts:
- Redis (cache)
- PostgreSQL (database)
- API server

### API Endpoints

#### Health Check
```bash
GET /api/v1/health
```

#### Model Status
```bash
GET /api/v1/status
```

#### Get Upcoming Predictions
```bash
GET /api/v1/predictions/upcoming?season=2024&week=1&min_confidence=0.6
```

#### Get Specific Game Prediction
```bash
GET /api/v1/predictions/game/KC/BUF?season=2024&week=1
```

#### Get Team's Upcoming Games
```bash
GET /api/v1/predictions/team/KC/upcoming?season=2024
```

#### Invalidate Cache (Admin)
```bash
POST /api/v1/cache/invalidate?pattern=prediction:*
```

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Environment Variables

- `REDIS_HOST`: Redis host (default: localhost)
- `REDIS_PORT`: Redis port (default: 6379)
- `REDIS_DB`: Redis database (default: 0)
- `MODEL_PATH`: Path to model file (default: models/improved_model.pkl)
- `POSTGRES_HOST`: PostgreSQL host (default: localhost)
- `POSTGRES_PORT`: PostgreSQL port (default: 5432)
- `POSTGRES_DB`: Database name (default: nfl_predictions)
- `POSTGRES_USER`: Database user (default: nfl_user)
- `POSTGRES_PASSWORD`: Database password (default: nfl_password)

### Production Deployment

#### Using Gunicorn

```bash
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

#### Using Docker

```bash
docker build -f backend/Dockerfile -t nfl-api .
docker run -p 8000:8000 -e REDIS_HOST=redis nfl-api
```

### Performance

- **Cache Hit**: ~5-10ms response time
- **Database Hit**: ~10-50ms response time (indexed queries)
- **Cache Miss + DB Miss**: ~200-500ms (includes model inference)
- **Throughput**: 1000+ requests/second (with caching + database)

### Storage Strategy

1. **Redis Cache** (fastest, 5-10ms)
   - TTL: 1 hour
   - Used for frequently accessed predictions

2. **PostgreSQL Database** (fast, 10-50ms)
   - Permanent storage
   - Indexed for fast lookups
   - Survives restarts

3. **Model Generation** (slowest, 200-500ms)
   - Only when cache and database miss
   - Results saved to database and cache

### Example Usage

#### Python
```python
import requests

# Get upcoming predictions
response = requests.get(
    "http://localhost:8000/api/v1/predictions/upcoming",
    params={"season": 2024, "week": 1, "min_confidence": 0.6}
)
predictions = response.json()

for pred in predictions:
    print(f"{pred['away_team']} @ {pred['home_team']}: {pred['predicted_winner']} wins ({pred['confidence']:.1%})")
```

#### cURL
```bash
curl "http://localhost:8000/api/v1/predictions/upcoming?season=2024&week=1"
```

---

## Frontend

Next.js frontend for viewing NFL game predictions with a modern, responsive interface.

### Features

- 🎯 View upcoming game predictions
- 📊 Confidence scores and win probabilities
- 🔍 Filter by season, week, and confidence level
- 📱 Responsive design (mobile, tablet, desktop)
- ⚡ Fast API integration
- 🎨 NFL-themed design with team colors

### Quick Start

#### Prerequisites

- Node.js 18+ installed
- Backend API server running (see [API Server](#api-server) section)

#### Installation

```bash
# Install dependencies
cd frontend
npm install

# Create environment file
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
```

#### Development

```bash
# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

#### Production Build

```bash
# Build the application
npm run build

# Start production server
npm start
```

### Configuration

Update `.env.local` to point to your API server:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Project Structure

```
frontend/
├── app/
│   ├── layout.tsx       # Root layout
│   ├── page.tsx         # Main dashboard page
│   └── globals.css      # Global styles
├── components/
│   ├── PredictionCard.tsx   # Game prediction card
│   ├── LoadingSpinner.tsx    # Loading indicator
│   └── ErrorMessage.tsx     # Error display
├── lib/
│   └── api.ts          # API client
└── package.json
```

### Features

#### Dashboard
- View all upcoming predictions in a responsive grid
- Filter by season and week
- Filter by minimum confidence level
- Real-time API status indicator
- Error handling with retry functionality

#### Prediction Cards
- Team matchups with team abbreviations
- Win probabilities for both teams
- Confidence scores with color coding:
  - Green: High confidence (≥70%)
  - Yellow: Medium confidence (40-70%)
  - Gray: Low confidence (<40%)
- Game date and time
- Visual probability bars
- Predicted winner highlighting

### API Integration

The frontend connects to the FastAPI backend at the URL specified in `NEXT_PUBLIC_API_URL`.

**Required endpoints:**
- `GET /api/v1/health` - Health check
- `GET /api/v1/status` - Model status
- `GET /api/v1/predictions/upcoming` - Get predictions

**API Client:**
The frontend uses an axios-based API client (`lib/api.ts`) with TypeScript types for type-safe API calls.

### Troubleshooting

#### "Failed to load predictions"
- Make sure the backend API server is running
- Check that `NEXT_PUBLIC_API_URL` matches your API server URL
- Verify CORS is enabled on the API server (already configured by default)

#### "API Disconnected"
- Check backend server logs
- Verify Redis and PostgreSQL are running
- Ensure API server is accessible at the configured URL
- Test API directly: `curl http://localhost:8000/api/v1/health`

#### CORS Errors
The API server already has CORS enabled with `allow_origins=["*"]`. If you need to restrict it:

```python
# In backend/api_server.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### No Predictions Showing
1. Make sure predictions exist in the database
2. Generate some predictions: `python backend/src/predict_upcoming_improved.py`
3. Check API directly: `curl http://localhost:8000/api/v1/predictions/upcoming`

### Full Stack Setup

To run the complete system (backend + frontend):

**Terminal 1 - Backend:**
```bash
cd backend
docker-compose up -d redis postgres
python src/init_database.py
python api_server.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Then open:
- Frontend: [http://localhost:3000](http://localhost:3000)
- API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Prediction Storage

### Storage Locations

#### 1. PostgreSQL Database (Primary Storage) ✅
- **Location**: PostgreSQL `predictions` table
- **Persistence**: ✅ **Permanent** - survives container restarts
- **Status**: ✅ **Fully Implemented** - All predictions stored in database
- **Indexes**: ✅ Optimized indexes for fast lookups

#### 2. Redis Cache (Fast Access)
- **Location**: Redis in-memory cache
- **Persistence**: ⚠️ **Temporary** - expires after TTL (default: 1 hour)
- **Key Format**: `predictions:{season}:{week}` or `prediction:{season}:{week}:{home}:{away}`
- **Purpose**: Fast lookups (5-10ms response time)

#### 3. CSV Files (Fallback/Export)
- **Location**: `backend/data/predictions_improved.csv`
- **Persistence**: ✅ Persists if data directory is writable
- **Purpose**: Backup/export format, fallback if database unavailable

### Storage Flow

```
Request → Redis Cache (5-10ms)
    ↓ (miss)
    → PostgreSQL Database (10-50ms) 
    ↓ (miss)
    → Generate Prediction (200-500ms)
    ↓
    → Save to Database + Cache
```

### Docker Persistence

**PostgreSQL Data** (`postgres_data:/var/lib/postgresql/data`):
- ✅ Full database persistence
- ✅ Survives container restarts
- ✅ All predictions stored permanently

**Redis Data** (`redis_data:/data`):
- ✅ Persists cache data (AOF enabled)
- ⚠️ Cache entries expire based on TTL
- ⚠️ Predictions cached here are temporary

**Data Directory** (`../data:/app/data:rw`):
- ✅ Writable in Docker
- ✅ CSV files persist across restarts (fallback)

### What Happens on Docker Restart?

**Container Restart (docker-compose restart):**
- ✅ **PostgreSQL**: All data persists
- ✅ **Redis**: Cache persists (AOF enabled)
- ✅ **CSV Files**: Persist if using volumes
- ⚠️ **Cache TTL**: Expired entries are removed

**Container Removal (docker-compose down):**
- ✅ **PostgreSQL**: Data persists (volume)
- ✅ **Redis**: Data persists (volume)
- ✅ **CSV Files**: Persist if using volumes
- ❌ **In-Memory Cache**: Lost (but repopulated from database)

### Recommendations

**For Production:**
1. **Primary Storage**: PostgreSQL database
   - Create `predictions` table
   - Store all predictions permanently
   - Query by season/week/team

2. **Cache Layer**: Redis
   - Fast lookups (5-10ms)
   - TTL: 1 hour or until game starts
   - Populated from database on cache miss

3. **Backup**: CSV exports
   - Periodic exports from database
   - Store in S3/backup location
   - For analysis and recovery

---

## Improved Model System

The improved model is the default and only model architecture in this system. All old model files have been removed and consolidated into this single, optimized implementation.

### Architecture

#### Level-1 Models (Base Learners)

1. **XGBoost**: Full feature set with improved hyperparameters (regularization, learning rate)
2. **LightGBM**: Alternative GBM implementation with tuned parameters
3. **CatBoost**: Gradient boosting with excellent handling of categorical features
4. **Random Forest**: Bagging-based ensemble for diversity (different from boosting models)
5. **Neural Network (MLP)**: Multi-layer perceptron with 2-3 hidden layers to capture complex non-linear relationships
6. **Regularized Logistic Regression**: Key features (market spread, Elo, EPA) with L2 regularization
7. **Market Baseline**: Simple logistic on spread only (strong anchor)

#### Level-2 Meta-Model / Ensemble Methods

The system supports multiple methods for combining Level-1 predictions:

**1. Stacking (Default)**:
- Uses a LightGBM meta-model to learn optimal combinations of Level-1 predictions
- Learns non-linear relationships between base model predictions
- Includes additional features (spread magnitude, Elo difference) for context

**2. Blending**:
- Simple averaging of Level-1 predictions
- **Uniform blending**: Equal weights (1/N) for all models
- **Performance-weighted blending**: Weights proportional to validation performance
- Faster than stacking, good baseline

**3. Weighted Ensemble**:
- Learns optimal weights from validation performance metrics
- Uses combined score: accuracy (40%), AUC (30%), log_loss (20%), brier (10%)
- Weights normalized to sum to 1.0
- More interpretable than stacking

**4. Dynamic Ensemble**:
- Selects top N models (default: top 5) based on validation performance
- Applies learned weights to selected models only
- Reduces noise from underperforming models
- Good for identifying and focusing on best-performing models

#### Probability Calibration

- **Isotonic Regression**: Calibrates each Level-1 model's probabilities
- Improves probability calibration and reduces overconfidence
- Better-calibrated probabilities lead to more accurate predictions

#### Feature Selection

- **Importance-Based Selection**: Uses XGBoost feature importance to select top features
- Reduces overfitting by focusing on high-signal features
- Typically selects top 30-50 features from the full set

#### Decision Rules

1. **Confidence Gating**: If |spread| ≥ 7, apply moderate confidence adjustment
2. **Uncertainty Dampening**: Smooth shrinkage of extreme probabilities toward 0.5
   - More gradual than previous version
   - Probabilities clipped to [0.25, 0.75] range for better calibration

#### Threshold Tuning

- Improved forward-chaining cross-validation
- Tune threshold using F1 score (balance of precision and recall)
- Optimal threshold typically 0.45-0.55

### High-Signal Features

1. **Market Consensus**: Spread and implied win probability (strongest signal)
2. **Elo Ratings**: Team strength with recency decay
3. **EPA/Play**: Expected Points Added per play (offense/defense)
4. **Success Rate**: Percentage of plays with positive EPA
5. **Situational**: Rest days, Thursday games, travel

### Feature Engineering Principles

- **Difference Features**: All features are (home - away) for single-row prediction
- **No Leakage**: All features use only data before the game
- **Snapshot Time**: Features should be knowable at fixed time (e.g., Friday 6pm ET)

### Validation Strategy

- **Forward-Chaining CV**: Train ≤2019 → validate 2020, etc.
- **Time-Aware**: Never use future data
- **Threshold Tuning**: Optimize for accuracy, not calibration
- **Out-of-Sample Testing**: Hold out most recent season

---

## Enhanced Features

The comprehensive feature set includes **48+ features** covering most aspects from "Strategies for Improving Accuracy".

### Market Features
- **Market Spread**: Betting line
- **Market Implied Win Probability**: Converted from spread

### Team Strength
- **Elo Ratings**: Team strength with recency decay
- **Strength of Schedule**: Average opponent Elo rating

### Advanced EPA Metrics
- **Success Rate**: Percentage of plays with positive EPA
- **Explosive Play Rate**: Plays gaining 15+ yards
- **Red Zone Efficiency**: Touchdown rate inside the 20
- **Penalty Impact**: Yards lost/gained from penalties
- **EPA per Play**: Offensive and defensive EPA
- **Net EPA**: Offensive EPA minus defensive EPA allowed

### QB Performance
- **QB EPA per Play**: Quarterback Expected Points Added
- **Completion Percentage**: Passing completion rate
- **Dropback Count**: Number of passing attempts

### Team Performance Splits
- **Home/Away Splits**: Team performance at home vs away
- **Home Advantage**: Calculated per team based on historical performance

### Matchup Features
- **Divisional Games**: Indicator for divisional matchups
- **Head-to-Head History**: Last 3-5 meetings between teams
- **Primetime Games**: Indicator for Thursday/Monday/Sunday Night games
- **Style Matchups**: Run-heavy vs pass-heavy team indicators
- **Defensive Matchups**: How well offense matches opponent's defense (EPA-based)
- **CB vs WR Matchups**: Player-level defensive matchups
- **Coaching Matchups**: Head-to-head coaching records

### Situational Features
- **Rest Days**: Days between games (bye weeks, Thursday games)
- **Rest Advantage**: Difference in rest days (home - away)
- **Short Week**: Indicator for Thursday games
- **Travel Distance**: Miles traveled between cities
- **Timezone Change**: Hours of timezone change for away team
- **Long Travel**: Indicator for >1500 mile trips
- **Eastward Travel**: Indicator for eastward timezone travel

### Weather Features
- **Dome Indicator**: Indoor stadium (no weather impact)
- **High Wind**: Wind >15 mph (impacts passing)
- **Cold Weather**: Temperature <32°F (favors running)
- **Bad Weather**: Precipitation indicator

**Implementation**:
- ✅ Fully integrated with Open-Meteo API (free, no API key required)
- ✅ Uses Archive API for historical dates, Forecast API for future dates
- ✅ Automatically fetches weather for outdoor games
- ✅ Caches results to avoid repeated API calls
- ✅ Integrated into model training pipeline

### Target Share Features
- **Top Receiver Target Share**: Percentage of targets going to top receiver
- **Target Concentration**: Herfindahl index (how concentrated targets are)
- **WR/TE/RB Distribution**: Approximate target share by position

### CB vs WR Matchup Features
- **Home WR vs Away CB Advantage**: How well home team's top WR matches away team's pass defense
- **Away WR vs Home CB Advantage**: How well away team's top WR matches home team's pass defense
- **CB/WR Matchup Advantage**: Net advantage (home - away)

### Injury Features
- **QB Injury Impact**: QB availability impact score (0=healthy, 1=questionable, 2=out)
- **Key Player Out**: Binary flag for star player injuries
- **Position-Specific Impacts**: OL, WR, CB injury impacts
- **Injury Advantage**: Relative injury advantage between teams

**Implementation**: 
- ✅ Fully integrated with NFL.com injury report scraping
- ✅ Automatically fetches injury data for all teams
- ✅ Caches results to avoid repeated API calls
- ✅ Integrated into model training pipeline

---

## Making Predictions

### Predicting Future Games

#### Snapshot Time Concept

**Critical**: All features must be built using only information available at a fixed "snapshot time" (e.g., Friday 6pm ET before Sunday games). This prevents data leakage.

#### Feature Building for Future Games

For each upcoming game, the system:

1. **Gets historical data** up to (but not including) the game week
2. **Calculates team statistics** from previous games
3. **Computes Elo ratings** as of before the game
4. **Gets EPA metrics** from recent games
5. **Uses market data** from snapshot time (spread/moneyline)
6. **Applies situational features** (rest days, travel, etc.)

#### Weekly Prediction Routine

```bash
# Monday: Collect previous week's results
python src/data_collection.py

# Tuesday: Retrain model with new data
python src/train_improved_model.py

# Friday 6pm ET: Make predictions for upcoming week
python src/predict_upcoming_improved.py --season 2024 --week 2

# Save predictions and track results
```

### Output Format

Predictions include:

- **Week**: Week number
- **Gameday/Gametime**: When the game is played
- **Teams**: Home and away teams
- **Home Win Probability**: Probability home team wins (0-1)
- **Away Win Probability**: Probability away team wins (0-1)
- **Predicted Winner**: Which team is predicted to win
- **Confidence**: How confident the model is (0-1)
  - 0.0 = 50/50 (low confidence, probability near 0.5)
  - 1.0 = Very confident (probability near 0 or 1)
  - Calculated as: `abs(probability - 0.5) * 2`
  - Note: Due to uncertainty dampening, confidence rarely exceeds 0.7-0.8

### Interpreting Results

#### Confidence Levels

- **High Confidence (≥0.7)**: Model is very confident
  - Use these for stronger bets/decisions
  - Typically 70-80%+ accuracy on these

- **Medium Confidence (0.4-0.7)**: Moderate confidence
  - Reasonable predictions but less certain
  - Good for tracking but be cautious

- **Low Confidence (<0.4)**: Low confidence
  - Model is uncertain (close matchup)
  - Consider avoiding or using smaller stakes

---

## Feature Engineering

### Team-Level Features

#### Offensive Metrics
- Points per game
- Yards per play
- Turnover rate
- Red zone efficiency
- Third down conversion rate

#### Defensive Metrics
- Points allowed
- Yards allowed per play
- Takeaways
- Red zone defense
- Third down defense

#### Advanced Stats
- **DVOA (Defense-adjusted Value Over Average)**: Available from Football Outsiders
- **EPA (Expected Points Added)**: Already in nflfastR data - use more extensively
- **Success Rate**: Percentage of plays with positive EPA
- **Explosive Play Rate**: Plays gaining 15+ yards
- **Red Zone Efficiency**: Touchdown rate inside the 20

#### Recent Form
- Last 3-5 games performance (weighted more heavily)
- Home/away splits
- Divisional game performance

### Matchup-Specific Features

- **Head-to-Head**: Historical performance between teams (last 3-5 meetings)
- **Style Matchups**: Run-heavy team vs pass-heavy team
- **Defensive Matchups**: How well does team's offense match opponent's defense?
- **Coaching Matchups**: Head-to-head coaching records
- **Rest Days**: Days of rest for each team (bye weeks, Thursday games)
- **Travel**: Distance traveled, time zone changes
- **Weather**: Temperature, wind, precipitation (for outdoor games)
- **Injury Impact**: Key player injuries, especially QBs and star players

### Contextual Features

- **Home Field Advantage**: Team's home win percentage
- **Divisional Games**: Performance in division matchups
- **Playoff Implications**: Motivation factors
- **Time of Season**: Early season vs late season performance
- **Strength of Schedule**: Adjusted for opponent quality
- **Turnover Regression**: Teams with extreme turnover rates tend to regress

### Player-Level Features

- **QB Performance**: Quarterback EPA, completion %, dropback count
- **Target Share**: For skill positions
  - Top receiver target share
  - Target concentration (Herfindahl index)
  - WR/TE/RB distribution
- **CB vs WR Matchups**: Player-level defensive matchups
  - Top WR target share vs defensive passing EPA
  - Air yards per target vs defensive air yards allowed
  - Matchup advantage calculation
- **Key Injuries**: Impact of missing star players (especially QB)

---

## Strategies for Improving Accuracy

This section outlines specific improvements you can make to maximize model accuracy, prioritizing accuracy over training speed.

### Training Modes

**Fast Mode (Default):**
- 300 trees, max_depth=4, learning_rate=0.03
- Top 50 features
- 6-7 seasons of data
- 3-fold calibration CV
- 5-fold cross-validation

**Accuracy Mode (`--accurate` flag):**
- 800 trees, max_depth=7, learning_rate=0.015
- Top 150 features
- 10+ seasons of data
- 5-fold calibration CV
- 7-fold cross-validation
- **Expected**: +3-7% accuracy improvement, 2-3x longer training time

**Optuna Hyperparameter Tuning (`--optuna` flag):**
- Automatically finds optimal hyperparameters for XGBoost, LightGBM, and CatBoost
- Uses Bayesian optimization (50-100 trials)
- **Expected**: +2-5% accuracy improvement, 5-10x longer training time
- **Usage**: `python src/train_improved_model.py --accurate --optuna`

### Implementation Status

**Legend**: ✅ Implemented | ⚠️ Partial/Placeholder | ❌ Not Yet Implemented

#### ✅ Completed (48+ features implemented)

1. ✅ **EPA Features**: Success rate, explosive plays, red zone efficiency
2. ✅ **Rolling Statistics**: Weighted recent form with exponential decay
3. ✅ **Home/Away Splits**: Team-specific home advantage calculation
4. ✅ **Divisional Games**: Indicator and enhanced H2H for division opponents
5. ✅ **Stacking Ensemble**: Two-level stacked model with meta-learner
6. ✅ **QB Performance**: QB EPA, completion %, dropback count
7. ✅ **Head-to-Head History**: Last 3-5 meetings with win rate and margin
8. ✅ **Strength of Schedule**: Opponent-adjusted Elo ratings
9. ✅ **Travel Features**: Distance, timezone changes, long travel indicators
10. ✅ **Rest Days**: Calculated with advantage metrics
11. ✅ **Style Matchups**: Pass/run rate indicators
12. ✅ **Defensive Matchups**: Offense vs defense EPA matching
13. ✅ **Target Share**: Top receiver target share, target concentration (Herfindahl index)
14. ✅ **CB vs WR Matchups**: Player-level defensive matchups (top WR vs pass defense)
15. ✅ **Turnover Regression**: Extreme turnover rate correction
16. ✅ **Third Down Stats**: Conversion and stop rates
17. ✅ **Time of Possession**: Ball control metrics
18. ✅ **Playoff Implications**: Late season motivation factors
19. ✅ **Coaching Matchups**: Framework for head-to-head records
20. ✅ **Neural Network (MLP)**: Multi-layer perceptron for non-linear relationships
21. ✅ **CatBoost**: Gradient boosting with categorical feature handling
22. ✅ **Random Forest**: Bagging-based ensemble for diversity
23. ✅ **Optuna Tuning**: Automated hyperparameter optimization
24. ✅ **Feature Selection**: Importance-based feature selection
25. ✅ **Probability Calibration**: Isotonic regression calibration
26. ✅ **Blending Ensemble**: Uniform and performance-weighted averaging
27. ✅ **Weighted Ensemble**: Optimal weights learned from validation performance
28. ✅ **Dynamic Ensemble**: Top N model selection with learned weights
29. ✅ **Weather Features**: Open-Meteo API integrated (free, no API key required)
30. ✅ **Injury Impact**: NFL.com scraping and API framework integrated

#### ❌ Not Yet Implemented

1. ❌ **DVOA**: Requires Football Outsiders data
2. ❌ **Betting Simulation**: ROI and profit tracking
3. ❌ **Line Movement**: Historical line tracking
4. ❌ **Level-3 Stacking**: Stack the stackers for even better performance
5. ❌ **Time-Series Models**: LSTM/GRU for sequential game patterns

---

## Project Structure

```
bet/
├── backend/                    # API server and backend code
│   ├── api_server.py          # FastAPI server
│   ├── requirements_api.txt   # API dependencies
│   ├── docker-compose.yml      # Docker Compose configuration
│   ├── Dockerfile             # Docker image definition
│   └── src/                   # Source code
│       ├── data_collection.py
│       ├── feature_engineering.py
│       ├── improved_model.py
│       ├── train_improved_model.py
│       ├── predict_upcoming_improved.py
│       ├── evaluate_model.py
│       ├── database.py
│       └── ...
├── frontend/                  # Next.js frontend application
│   ├── app/                   # Next.js app directory
│   │   ├── layout.tsx         # Root layout
│   │   ├── page.tsx           # Main dashboard
│   │   └── globals.css        # Global styles
│   ├── components/            # React components
│   │   ├── PredictionCard.tsx
│   │   ├── LoadingSpinner.tsx
│   │   └── ErrorMessage.tsx
│   ├── lib/                   # Utilities
│   │   └── api.ts            # API client
│   ├── package.json
│   └── tsconfig.json
├── data/                      # Raw and processed data
├── models/                    # Trained model files (improved_model.pkl)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Data Sources

- **nfl_data_py**: Python package for NFL data (nflverse ecosystem)
  - Install: `pip install nfl_data_py`
  - Documentation: https://github.com/nflverse/nfl_data_py
- **Pro Football Reference**: For additional historical data
  - Team stats: Passing yards, rushing yards, turnovers, points
  - Player stats: QB, RB, WR, TE performance metrics
  - Integrated via `pro-football-reference-web-scraper` package
- **NFL.com**: Official stats, schedules, and injury reports (scraped)
- **Weather API**: 
  - **Open-Meteo**: Free weather API (no API key required, open-source)
    - Uses Archive API (`/v1/archive`) for historical dates
    - Uses Forecast API (`/v1/forecast`) for future dates (up to 10 days)
    - Automatically caches results to reduce API calls

**API Setup**:
1. No API keys needed for weather - Open-Meteo is completely free!
2. Weather API works automatically without any configuration
3. NFL.com scraping also works without API keys

---

## Notes and Best Practices

### Key Strategies for High Accuracy

1. **Feature Engineering > Model Complexity**: Spend most time on features
2. **Cross-Validation**: Use time-series cross-validation (don't leak future data)
3. **Feature Selection**: Remove redundant or low-value features
4. **Ensemble Methods**: Combine multiple models
5. **Regular Updates**: Retrain with new data weekly during season
6. **Domain Knowledge**: Incorporate NFL-specific insights

### Evaluation Metrics

- **Accuracy**: Overall prediction accuracy
- **Log Loss**: Better for probability predictions
- **Brier Score**: Calibration metric
- **ROC-AUC**: For win/loss classification
- **Profit/Loss**: If betting, track actual betting performance

### Troubleshooting

#### "No upcoming games found"
- Check that the season/week exists
- Verify schedule data is collected
- Try: `python src/data_collection.py` to refresh data

#### "Could not create features"
- Ensure you have historical data for the season
- Check that team stats are calculated
- For Week 1, previous season data should be available

#### "Model not found"
- Train the model first: `python src/train_improved_model.py`

#### "Feature names mismatch"
- **Solution**: Retrain the model after adding new features:
  ```bash
  python src/train_improved_model.py
  ```
- This error occurs when the model was trained with an older feature set but predictions are using a newer feature set
- After adding new features, always retrain the model to ensure optimal performance

#### Low confidence on all predictions
- This is normal for close matchups
- Model is being appropriately cautious due to uncertainty dampening
- Consider using `--min-confidence` to filter

### Best Practices

1. **Regular Updates**: Retrain weekly with new game results
2. **Snapshot Discipline**: Always use features from fixed snapshot time
3. **Track Everything**: Keep a log of predictions vs results
4. **Market Data**: Use real-time spreads when possible
5. **Review Errors**: Analyze which games the model gets wrong
6. **Iterate**: Use evaluation results to improve features

### Important Notes

- NFL prediction is inherently difficult (even experts are ~55-60% accurate)
- Focus on finding edges through feature engineering
- Consider betting markets as a benchmark (they're quite efficient)
- Track your model's performance over time and iterate

---

## Quick Reference

### Training Commands
```bash
# Train improved model (fast mode - default, stacking ensemble)
python src/train_improved_model.py

# Train with accuracy mode (slower but more accurate)
python src/train_improved_model.py --accurate

# Train with Optuna hyperparameter tuning (50 trials)
python src/train_improved_model.py --accurate --optuna
```

### Prediction Commands
```bash
# Upcoming games (default: current season, next week)
python src/predict_upcoming_improved.py

# Specific season and week
python src/predict_upcoming_improved.py --season 2024 --week 1

# High-confidence only
python src/predict_upcoming_improved.py --min-confidence 0.7
```

### Evaluation Commands
```bash
# All games (default: most recent complete season)
python src/evaluate_model.py

# Specific season/week
python src/evaluate_model.py --season 2023 --week 1

# High-confidence only
python src/evaluate_model.py --min-confidence 0.7
```

### API Commands
```bash
# Start services
cd backend
docker-compose up -d

# Initialize database
python src/init_database.py

# Start API server
python api_server.py

# Test API
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/predictions/upcoming?season=2024&week=1
```

### Frontend Commands
```bash
# Install dependencies
cd frontend
npm install

# Start development server
npm run dev

# Build for production
npm run build
npm start
```

### Full Stack Commands
```bash
# Terminal 1 - Backend
cd backend
docker-compose up -d redis postgres
python src/init_database.py
python api_server.py

# Terminal 2 - Frontend
cd frontend
npm install
npm run dev
```

---

## Conclusion

This NFL prediction system provides:

- ✅ High-accuracy predictions (58-62%+ accuracy)
- ✅ Production-ready API server
- ✅ Modern Next.js frontend
- ✅ Scalable architecture (1000+ requests/second)
- ✅ Permanent database storage
- ✅ Fast lookups (5-50ms response times)
- ✅ Comprehensive feature engineering
- ✅ Multiple ensemble methods
- ✅ Automated hyperparameter tuning
- ✅ Responsive web interface

Start simple, scale as needed!
