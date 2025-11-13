"""
Initialize Database Schema

Creates all tables and indexes for the NFL predictions database.
Run this once to set up the database schema.

Prerequisites:
- PostgreSQL must be running
- Environment variables set (or defaults will be used)
- Database must exist (created automatically in Docker)
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from database import init_db, Base, get_db_url

def main():
    """Initialize database schema"""
    print("=" * 60)
    print("NFL Predictions Database Initialization")
    print("=" * 60)
    
    # Show connection info
    db_url = get_db_url()
    # Mask password in display
    display_url = db_url.split('@')[0].split(':')[:-1]
    display_url.append('***')
    display_url = ':'.join(display_url) + '@' + '@'.join(db_url.split('@')[1:])
    print(f"\nConnecting to: {display_url}")
    
    try:
        print("\nInitializing database schema...")
        engine = init_db(create_tables=True)
        
        print("\n✓ Database connection successful")
        print("✓ Database tables created")
        print("✓ Indexes created for optimized lookups")
        
        print("\n" + "=" * 60)
        print("Tables created:")
        print("=" * 60)
        for table in Base.metadata.tables:
            print(f"  - {table}")
        
        print("\n" + "=" * 60)
        print("Indexes created (for hot lookups):")
        print("=" * 60)
        for table_name, table in Base.metadata.tables.items():
            for index in table.indexes:
                cols = ', '.join([c.name for c in index.columns])
                print(f"  - {table_name}.{index.name}: ({cols})")
        
        print("\n" + "=" * 60)
        print("✓ Database initialization complete!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ Database initialization failed")
        print("=" * 60)
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure PostgreSQL is running:")
        print("   docker-compose up postgres")
        print("   OR")
        print("   docker run -d -p 5432:5432 -e POSTGRES_DB=nfl_predictions \\")
        print("     -e POSTGRES_USER=nfl_user -e POSTGRES_PASSWORD=nfl_password \\")
        print("     postgres:15-alpine")
        print("\n2. Check environment variables:")
        print(f"   POSTGRES_HOST: {os.getenv('POSTGRES_HOST', 'localhost')}")
        print(f"   POSTGRES_PORT: {os.getenv('POSTGRES_PORT', '5432')}")
        print(f"   POSTGRES_DB: {os.getenv('POSTGRES_DB', 'nfl_predictions')}")
        print(f"   POSTGRES_USER: {os.getenv('POSTGRES_USER', 'nfl_user')}")
        print("\n3. Verify connection:")
        print("   docker-compose exec postgres psql -U nfl_user -d nfl_predictions -c 'SELECT 1;'")
        sys.exit(1)

if __name__ == "__main__":
    main()

