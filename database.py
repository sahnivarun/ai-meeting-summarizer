# database.py
import sqlite3
import datetime # Make sure datetime is imported

DATABASE_NAME = 'meetings.db'

def get_db_connection():
    # Enable detect_types to parse declared types (like TIMESTAMP)
    conn = sqlite3.connect(DATABASE_NAME, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    # PARSE_COLNAMES allows using type hints in column names like "foo [timestamp]"
    # For this project, PARSE_DECLTYPES is the primary one needed for TIMESTAMP columns.
    conn.row_factory = sqlite3.Row # Allows accessing columns by name
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS meetings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- This is key for detect_types
        transcript TEXT,
        summary TEXT,
        processing_status TEXT DEFAULT 'pending' -- pending, transcribing, processing_nlp, completed, error
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS action_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meeting_id INTEGER,
        task TEXT NOT NULL,
        owner TEXT,
        due_date TEXT, -- Storing as text for simplicity, LLM will provide this
        status TEXT DEFAULT 'pending', -- pending, completed
        FOREIGN KEY (meeting_id) REFERENCES meetings (id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meeting_id INTEGER,
        decision_text TEXT NOT NULL,
        FOREIGN KEY (meeting_id) REFERENCES meetings (id)
    )
    ''')

    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    print("Database initialized.")
