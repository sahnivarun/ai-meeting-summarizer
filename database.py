# database.py
import sqlite3
import datetime
import logging

DATABASE_NAME = 'meetings.db'
logger = logging.getLogger(__name__)

def adapt_datetime_iso(val):
    if val is None: return None
    return val.isoformat()

def convert_timestamp(val):
    if val is None: return None
    try:
        decoded_val = val.decode()
        if '.' in decoded_val:
            return datetime.datetime.fromisoformat(decoded_val.replace(' ', 'T', 1) if 'T' not in decoded_val else decoded_val)
        else:
            return datetime.datetime.strptime(decoded_val, '%Y-%m-%d %H:%M:%S' if 'T' not in decoded_val else '%Y-%m-%dT%H:%M:%S')
    except ValueError as e:
        logger.error(f"Error converting timestamp string '{val.decode()}' to datetime: {e}")
        return None

sqlite3.register_adapter(datetime.datetime, adapt_datetime_iso)
sqlite3.register_converter("timestamp", convert_timestamp)

def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn

def _add_column_if_not_exists(cursor, table_name, column_name, column_type_with_default="TEXT"):
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [info[1] for info in cursor.fetchall()]
    if column_name not in columns:
        logger.info(f"Adding '{column_name}' column to '{table_name}' table with type '{column_type_with_default}'.")
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type_with_default}")
        return True
    return False

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS meetings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT, -- <<< ALLOWS NULL for text-based inputs
        upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        transcript TEXT,
        summary TEXT,
        processing_status TEXT DEFAULT 'pending',
        meeting_title TEXT, 
        scheduled_datetime TIMESTAMP, 
        end_datetime TIMESTAMP,       
        agenda TEXT,                  
        attendees TEXT                
    )
    ''')
    _add_column_if_not_exists(cursor, "meetings", "meeting_title", "TEXT")
    _add_column_if_not_exists(cursor, "meetings", "scheduled_datetime", "TIMESTAMP")
    _add_column_if_not_exists(cursor, "meetings", "end_datetime", "TIMESTAMP")
    _add_column_if_not_exists(cursor, "meetings", "agenda", "TEXT")
    _add_column_if_not_exists(cursor, "meetings", "attendees", "TEXT")

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
    _add_column_if_not_exists(cursor, "decisions", "status", "TEXT DEFAULT 'open'")
    _add_column_if_not_exists(cursor, "decisions", "resolution_notes", "TEXT")
        
    conn.commit()
    conn.close()
    logger.info("Database schema initialized/verified successfully.")

if __name__ == '__main__':
    if not logging.getLogger().hasHandlers(): 
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Initializing database directly from database.py script...")
    init_db()
    logger.info("Database initialization script finished.")