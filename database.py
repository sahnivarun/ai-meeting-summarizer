# database.py
import sqlite3
import datetime
import logging # <--- THIS WAS MISSING

DATABASE_NAME = 'meetings.db'

# --- For Python 3.12+ DeprecationWarning regarding default timestamp converter ---
def adapt_datetime_iso(val):
    """Adapt datetime.datetime to timezone-naive ISO 8601 date."""
    return val.isoformat()

def convert_timestamp(val):
    """Convert ISO 8601 string to datetime.datetime object."""
    return datetime.datetime.fromisoformat(val.decode())

sqlite3.register_adapter(datetime.datetime, adapt_datetime_iso)
sqlite3.register_converter("timestamp", convert_timestamp)
# --- End Timestamp Converter ---

def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn

# Get a logger instance for messages from this module, especially init_db
# This logger will be named 'database' if this file is imported, or '__main__.database_init' if run directly.
# It's better to define it once at the module level if used in multiple functions here.
logger = logging.getLogger(__name__)

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS meetings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL, 
        upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        transcript TEXT,
        summary TEXT,
        processing_status TEXT DEFAULT 'pending'
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS action_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meeting_id INTEGER,
        task TEXT NOT NULL,
        owner TEXT,
        due_date TEXT,
        status TEXT DEFAULT 'pending', 
        FOREIGN KEY (meeting_id) REFERENCES meetings (id) ON DELETE CASCADE 
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meeting_id INTEGER,
        decision_text TEXT NOT NULL,
        status TEXT DEFAULT 'open', 
        resolution_notes TEXT,      
        FOREIGN KEY (meeting_id) REFERENCES meetings (id) ON DELETE CASCADE
    )
    ''')

    # Add new columns to decisions table if they don't exist (for existing DBs)
    # Check if 'status' column exists
    cursor.execute("PRAGMA table_info(decisions)")
    columns = [info[1] for info in cursor.fetchall()]
    
    if 'status' not in columns:
        logger.info("Adding 'status' column to 'decisions' table.")
        cursor.execute("ALTER TABLE decisions ADD COLUMN status TEXT DEFAULT 'open'")
    
    if 'resolution_notes' not in columns:
        logger.info("Adding 'resolution_notes' column to 'decisions' table.")
        cursor.execute("ALTER TABLE decisions ADD COLUMN resolution_notes TEXT")
        
    conn.commit()
    conn.close()
    logger.info("Database schema initialized/verified successfully.")


if __name__ == '__main__':
    # Configure basic logging if this script is run directly
    # This ensures that the logger messages from init_db are visible
    if not logging.getLogger().hasHandlers(): # Check if root logger already has handlers
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("Initializing database directly from database.py script...")
    init_db()
    logger.info("Database initialization script finished.")