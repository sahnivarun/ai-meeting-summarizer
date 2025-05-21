# database.py
import sqlite3
import datetime
import logging # Ensure logging is imported

DATABASE_NAME = 'meetings.db'
logger = logging.getLogger(__name__) # Define logger at module level

# --- For Python 3.12+ DeprecationWarning regarding default timestamp converter ---
def adapt_datetime_iso(val):
    """Adapt datetime.datetime to timezone-naive ISO 8601 date."""
    if val is None:
        return None
    return val.isoformat()

def convert_timestamp(val):
    """Convert ISO 8601 string to datetime.datetime object."""
    if val is None:
        return None
    try:
        # Handle both space and 'T' separator, and optional microseconds
        # Common formats: 'YYYY-MM-DD HH:MM:SS.ffffff', 'YYYY-MM-DD HH:MM:SS', 
        #                 'YYYY-MM-DDTHH:MM:SS.ffffff', 'YYYY-MM-DDTHH:MM:SS'
        decoded_val = val.decode()
        if '.' in decoded_val:
            if 'T' in decoded_val:
                return datetime.datetime.fromisoformat(decoded_val)
            else: # Replace space with T for fromisoformat if microseconds are present
                return datetime.datetime.fromisoformat(decoded_val.replace(' ', 'T', 1))
        else: # No microseconds
            if 'T' in decoded_val:
                return datetime.datetime.fromisoformat(decoded_val)
            else:
                 return datetime.datetime.strptime(decoded_val, '%Y-%m-%d %H:%M:%S')

    except ValueError as e:
        logger.error(f"Error converting timestamp string '{val.decode()}' to datetime: {e}")
        return None # Or raise error, or return original string if preferred

sqlite3.register_adapter(datetime.datetime, adapt_datetime_iso)
sqlite3.register_converter("timestamp", convert_timestamp) # "timestamp" must match column type in DDL
# --- End Timestamp Converter ---

def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn

def _add_column_if_not_exists(cursor, table_name, column_name, column_type_with_default="TEXT"):
    """Helper to add a column if it doesn't exist."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [info[1] for info in cursor.fetchall()]
    if column_name not in columns:
        logger.info(f"Adding '{column_name}' column to '{table_name}' table.")
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type_with_default}")
        return True
    return False

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # --- Meetings Table ---
    # filename: actual name of the audio file stored on disk (e.g., timestamped_original.mp3)
    # meeting_title: User-provided title, or a generated one like "Meeting X" or from summary.
    # upload_time: When the audio was uploaded/recorded and processing started.
    # scheduled_datetime, end_datetime, agenda, attendees: Kept for potential future use or old data, but will be NULL for new non-scheduled meetings.
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS meetings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT, 
        upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        transcript TEXT,
        summary TEXT,
        processing_status TEXT DEFAULT 'pending',
        meeting_title TEXT,      -- User-provided or generated title
        scheduled_datetime TIMESTAMP, -- Will be NULL for non-scheduled
        end_datetime TIMESTAMP,       -- Will be NULL for non-scheduled
        agenda TEXT,                  -- Will be NULL for non-scheduled
        attendees TEXT                -- Will be NULL for non-scheduled
    )
    ''')
    # Add columns to meetings table if they don't exist (for existing DBs)
    _add_column_if_not_exists(cursor, "meetings", "meeting_title", "TEXT")
    _add_column_if_not_exists(cursor, "meetings", "scheduled_datetime", "TIMESTAMP")
    _add_column_if_not_exists(cursor, "meetings", "end_datetime", "TIMESTAMP")
    _add_column_if_not_exists(cursor, "meetings", "agenda", "TEXT")
    _add_column_if_not_exists(cursor, "meetings", "attendees", "TEXT")
    # Ensure filename is not NOT NULL if it can be NULL for purely scheduled meetings (though we removed scheduling)
    # For safety, let's keep it as TEXT (allowing NULL) if your app handles that case.
    # If all meetings *must* have an audio file, then filename TEXT NOT NULL is fine.
    # Given your current app.py (no scheduling), meetings are only created with an audio file,
    # so filename TEXT NOT NULL (as in your provided snippet) is okay for new data.
    # If you had old scheduled data with NULL filename, the `ALTER TABLE` wouldn't add NOT NULL constraint.
    # Let's ensure filename in the CREATE TABLE is TEXT (allowing NULL for extreme flexibility)
    # and your app.py will always provide it for processed meetings.

    # --- Action Items Table ---
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS action_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meeting_id INTEGER,
        task TEXT NOT NULL,
        owner TEXT,
        due_date TEXT, -- Storing as text for simplicity
        status TEXT DEFAULT 'pending', -- 'pending', 'completed'
        FOREIGN KEY (meeting_id) REFERENCES meetings (id) ON DELETE CASCADE 
    )
    ''')

    # --- Decisions Table ---
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meeting_id INTEGER,
        decision_text TEXT NOT NULL,
        status TEXT DEFAULT 'open',      -- e.g., 'open', 'implemented', 'deferred'
        resolution_notes TEXT,           -- Notes on how/why the decision was resolved/status changed
        FOREIGN KEY (meeting_id) REFERENCES meetings (id) ON DELETE CASCADE
    )
    ''')
    # Add new columns to decisions table if they don't exist
    _add_column_if_not_exists(cursor, "decisions", "status", "TEXT DEFAULT 'open'")
    _add_column_if_not_exists(cursor, "decisions", "resolution_notes", "TEXT")
        
    conn.commit()
    conn.close()
    logger.info("Database schema initialized/verified successfully.")


if __name__ == '__main__':
    # Configure basic logging if this script is run directly
    if not logging.getLogger().hasHandlers(): 
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("Initializing database directly from database.py script...")
    init_db()
    logger.info("Database initialization script finished.")