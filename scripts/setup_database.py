#!/usr/bin/env python3
"""
SignalForge Database Setup Script

Initializes PostgreSQL database with SignalForge schema.

Usage:
    python scripts/setup_database.py --user postgres --password YOUR_PASSWORD
    python scripts/setup_database.py  # Will prompt for password
"""

import sys
import argparse
import getpass
from pathlib import Path

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")


class DatabaseSetup:
    """Initialize SignalForge database and schema."""
    
    def __init__(self, host: str = "localhost", port: int = 5432, 
                 user: str = "postgres", password: str = None, 
                 database: str = "signalforge"):
        self.host = host
        self.port = port
        self.user = user
        self.password = password or getpass.getpass(f"Password for {user}: ")
        self.database = database
        self.conn = None
        
    def connect_to_postgres(self):
        """Connect to default postgres database."""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database="postgres"
            )
            self.conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            logger.success(f"Connected to PostgreSQL at {self.host}:{self.port}")
            return True
        except psycopg2.OperationalError as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def create_database(self):
        """Create signalforge database if it doesn't exist."""
        cursor = self.conn.cursor()
        
        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
            (self.database,)
        )
        
        if cursor.fetchone():
            logger.info(f"Database '{self.database}' already exists")
        else:
            logger.info(f"Creating database '{self.database}'...")
            cursor.execute(f'CREATE DATABASE {self.database}')
            logger.success(f"Created database '{self.database}'")
        
        cursor.close()
    
    def create_user(self, user_password: str = None):
        """Create signalforge user if it doesn't exist."""
        cursor = self.conn.cursor()
        
        # Check if user exists
        cursor.execute(
            "SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = %s",
            ("signalforge_user",)
        )
        
        if cursor.fetchone():
            logger.info("User 'signalforge_user' already exists")
        else:
            logger.info("Creating user 'signalforge_user'...")
            # Use provided password or default
            password = user_password or "SignalForge2024!"
            cursor.execute(f"CREATE USER signalforge_user WITH PASSWORD '{password}'")
            logger.success(f"Created user 'signalforge_user' (password: {password})")
        
        cursor.close()
    
    def run_schema(self):
        """Execute schema SQL file."""
        # Reconnect to signalforge database
        self.conn.close()
        self.conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database
        )
        
        schema_path = Path(__file__).parent.parent / "infrastructure" / "sql" / "01_schema.sql"
        
        if not schema_path.exists():
            logger.error(f"Schema file not found: {schema_path}")
            return False
        
        logger.info(f"Running schema from {schema_path}...")
        
        with open(schema_path, 'r') as f:
            sql = f.read()
        
        cursor = self.conn.cursor()
        
        try:
            cursor.execute(sql)
            self.conn.commit()
            logger.success("Schema created successfully")
            return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to create schema: {e}")
            return False
        finally:
            cursor.close()
    
    def verify_setup(self):
        """Verify database setup."""
        cursor = self.conn.cursor()
        
        # Check schemas
        cursor.execute("""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name IN ('raw', 'features', 'models', 'analytics')
            ORDER BY schema_name
        """)
        
        schemas = [row[0] for row in cursor.fetchall()]
        logger.info(f"Created schemas: {', '.join(schemas)}")
        
        # Check tables
        cursor.execute("""
            SELECT table_schema, COUNT(*)
            FROM information_schema.tables
            WHERE table_schema IN ('raw', 'features', 'models', 'analytics')
            GROUP BY table_schema
            ORDER BY table_schema
        """)
        
        logger.info("Table counts:")
        for schema, count in cursor.fetchall():
            logger.info(f"  {schema}: {count} tables")
        
        cursor.close()
        logger.success("Database setup verified!")
    
    def run(self):
        """Run full setup."""
        logger.info("=" * 60)
        logger.info("SignalForge Database Setup")
        logger.info("=" * 60)
        
        if not self.connect_to_postgres():
            return False
        
        self.create_database()
        self.create_user()
        
        if not self.run_schema():
            return False
        
        self.verify_setup()
        
        logger.success("Database setup complete!")
        return True
    
    def close(self):
        """Close connection."""
        if self.conn:
            self.conn.close()


def main():
    parser = argparse.ArgumentParser(description="Setup SignalForge database")
    parser.add_argument("--host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--user", default="postgres", help="PostgreSQL admin user")
    parser.add_argument("--password", help="PostgreSQL admin password (will prompt if not provided)")
    parser.add_argument("--database", default="signalforge", help="Database name")
    
    args = parser.parse_args()
    
    setup = DatabaseSetup(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database
    )
    
    try:
        success = setup.run()
        sys.exit(0 if success else 1)
    finally:
        setup.close()


if __name__ == "__main__":
    main()
