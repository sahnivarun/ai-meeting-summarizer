# app.py
import os
import sqlite3
import logging # For logging
import openai # Import openai here to catch specific errors like AuthenticationError

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, g
from werkzeug.utils import secure_filename
from datetime import datetime # Ensure datetime is imported from datetime module
from ics import Calendar, Event
import dateparser

# Custom modules
from database import get_db_connection, init_db
from transcription import transcribe_audio, load_whisper_model
# nlp_processor functions are now more robust and return error strings
from nlp_processor import generate_summary, extract_action_items, extract_decisions

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'mp4', 'ogg', 'flac', 'webm'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB upload limit

# --- Logging Setup ---
# The basicConfig in nlp_processor.py likely configures the root logger.
# We can get a specific logger for this module ('app') for clarity if desired,
# or just use the root logger's configuration.
# Using logging.getLogger(__name__) is standard practice.
logger = logging.getLogger(__name__) # This will be 'app'
# If nlp_processor.py's basicConfig is effective, this logger will inherit its settings.
# If you want to ensure this logger has a specific level, you can set it:
# logger.setLevel(logging.INFO) # Or logging.DEBUG for more verbosity from this file

# --- End Logging Setup ---


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

with app.app_context():
    init_db()
    logger.info("Database initialized.")
    load_whisper_model() # Whisper model loading is logged within this function

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = get_db_connection()
    return db

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'audio_file' not in request.files:
            logger.warning("File upload attempt with no file part in request.")
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['audio_file']
        if file.filename == '':
            logger.warning("File upload attempt with no selected file.")
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp_prefix = datetime.now().strftime("%Y%m%d%H%M%S")
            unique_filename = f"{timestamp_prefix}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            meeting_id = None # Initialize meeting_id
            try:
                if not os.path.exists(app.config['UPLOAD_FOLDER']):
                    os.makedirs(app.config['UPLOAD_FOLDER'])
                    logger.info(f"Created upload folder: {app.config['UPLOAD_FOLDER']}")
                file.save(filepath)
                logger.info(f"File {unique_filename} saved to {filepath}")

                db = get_db()
                cursor = db.cursor()
                cursor.execute("INSERT INTO meetings (filename, processing_status) VALUES (?, ?)", 
                               (unique_filename, 'uploaded'))
                meeting_id = cursor.lastrowid
                db.commit()
                logger.info(f"Meeting record created with ID: {meeting_id}, status: uploaded.")
                
                # --- Transcription Stage ---
                cursor.execute("UPDATE meetings SET processing_status = ? WHERE id = ?", ('transcribing', meeting_id))
                db.commit()
                logger.info(f"Status updated to 'transcribing' for meeting ID {meeting_id} ({unique_filename}).")
                transcript_text = transcribe_audio(filepath) # This function has its own logging

                if not transcript_text: # transcribe_audio returns None or empty on failure
                    logger.error(f"Transcription failed for meeting ID {meeting_id} ({unique_filename}).")
                    cursor.execute("UPDATE meetings SET processing_status = ?, transcript = ? WHERE id = ?", 
                                   ('error', 'Transcription failed.', meeting_id))
                    db.commit()
                    flash(f'Transcription failed for {unique_filename}. Check server logs.', 'danger')
                    return redirect(url_for('index'))
                
                logger.info(f"Transcription complete for meeting ID {meeting_id}. Transcript length: {len(transcript_text)} chars.")
                cursor.execute("UPDATE meetings SET transcript = ?, processing_status = ? WHERE id = ?", 
                               (transcript_text, 'processing_nlp', meeting_id))
                db.commit()
                logger.info(f"Status updated to 'processing_nlp' for meeting ID {meeting_id}.")

                # --- NLP Processing Stage ---
                logger.info(f"Starting NLP processing for meeting ID {meeting_id}...")
                summary_result = generate_summary(transcript_text) # Returns summary or "ERROR:..." string
                logger.info(f"Received summary_result (len: {len(summary_result)}): '{summary_result[:100]}...' for meeting ID {meeting_id}")
                
                action_items_data = extract_action_items(transcript_text) # Returns list or empty list
                logger.info(f"Received {len(action_items_data)} action items for meeting ID {meeting_id}")
                
                decisions_data = extract_decisions(transcript_text) # Returns list or empty list
                logger.info(f"Received {len(decisions_data)} decisions for meeting ID {meeting_id}")

                nlp_error_occurred = False
                if summary_result.startswith("ERROR:"):
                    logger.error(f"NLP error during summary generation for meeting ID {meeting_id}: {summary_result}")
                    flash(f"NLP Error (Summary): {summary_result.replace('ERROR: ', '')}", 'warning')
                    nlp_error_occurred = True
                
                # Additional logging for empty action items/decisions if an NLP error occurred
                if nlp_error_occurred:
                    if not action_items_data:
                         logger.warning(f"Action items list is empty for meeting ID {meeting_id}, possibly due to NLP error: {summary_result}")
                    if not decisions_data:
                         logger.warning(f"Decisions list is empty for meeting ID {meeting_id}, possibly due to NLP error: {summary_result}")
                
                current_db_status = 'error' if nlp_error_occurred else 'completed'
                db_summary_to_store = summary_result # This is now guaranteed to be a string

                logger.info(f"Updating meeting ID {meeting_id} with final summary and status '{current_db_status}'.")
                cursor.execute("UPDATE meetings SET summary = ?, processing_status = ? WHERE id = ?", 
                               (db_summary_to_store, current_db_status, meeting_id))
                
                if not nlp_error_occurred: # Only insert if main NLP (summary) didn't report a critical error
                    logger.info(f"Attempting to insert {len(action_items_data)} action items for meeting ID {meeting_id}.")
                    for item in action_items_data:
                        cursor.execute("INSERT INTO action_items (meeting_id, task, owner, due_date) VALUES (?, ?, ?, ?)",
                                       (meeting_id, item.get('task'), item.get('owner'), item.get('due_date')))
                    
                    logger.info(f"Attempting to insert {len(decisions_data)} decisions for meeting ID {meeting_id}.")
                    for decision_text in decisions_data:
                        cursor.execute("INSERT INTO decisions (meeting_id, decision_text) VALUES (?, ?)",
                                       (meeting_id, decision_text))
                else:
                    logger.warning(f"Skipping insertion of action items/decisions for meeting ID {meeting_id} due to nlp_error_occurred=True.")
                
                db.commit()
                logger.info(f"Database commit successful for meeting ID {meeting_id} NLP results.")

                flash_message = f'File {unique_filename} processed.'
                flash_category = 'success'
                if nlp_error_occurred:
                    flash_message += " However, there were issues generating some NLP results (see details or logs)."
                    flash_category = 'warning'
                
                flash(flash_message, flash_category)
                return redirect(url_for('meeting_detail', meeting_id=meeting_id))

            # Specific Error Handling for NLP Stage
            except openai.AuthenticationError as e: # Raised if API key is invalid
                error_msg = f"OpenAI Authentication Error (check API Key): {e}"
                logger.critical(f"Meeting ID {meeting_id}: {error_msg}", exc_info=True)
                if meeting_id:
                    # Update DB even for this critical error before NLP functions are deeply called
                    db_conn_auth_err = get_db()
                    cur_auth_err = db_conn_auth_err.cursor()
                    cur_auth_err.execute("UPDATE meetings SET summary = ?, processing_status = ? WHERE id = ?", 
                                   (error_msg, 'error', meeting_id))
                    db_conn_auth_err.commit()
                flash(error_msg, 'danger')
                return redirect(url_for('index')) # Or request.url
            except Exception as e: # General catch-all for the processing block
                error_msg = f"Unexpected error processing file {unique_filename} (meeting ID {meeting_id}): {e}"
                logger.error(error_msg, exc_info=True)
                if meeting_id:
                    try:
                        db_conn_gen_err = get_db()
                        cur_gen_err = db_conn_gen_err.cursor()
                        # Store a truncated error message if it's very long
                        summary_error_text = f"Processing Error: {str(e)[:250]}" 
                        cur_gen_err.execute("UPDATE meetings SET summary = ?, processing_status = ? WHERE id = ?", 
                                       (summary_error_text, 'error', meeting_id))
                        db_conn_gen_err.commit()
                    except Exception as db_err:
                         logger.error(f"Failed to update meeting ID {meeting_id} status to error after general processing error: {db_err}")
                flash(f'An unexpected error occurred: {str(e)[:100]}... Check server logs.', 'danger') # Show truncated error
                return redirect(request.url)
        else: # File not allowed
            logger.warning(f"Upload attempt with disallowed file type: {file.filename}")
            flash('File type not allowed', 'warning')
            return redirect(request.url)

    # --- GET request part of index route ---
    db = get_db()
    cursor = db.cursor()
    # Order by upload_time as datetime objects correctly
    cursor.execute("SELECT id, filename, upload_time, processing_status, summary FROM meetings ORDER BY DATETIME(upload_time) DESC")
    meetings_raw = cursor.fetchall()
    
    meetings_list = []
    for m_raw in meetings_raw:
        m_item = dict(m_raw) 
        current_upload_time = m_item.get('upload_time')

        if isinstance(current_upload_time, str):
            parsed_time = None
            # Try parsing with microseconds, then without
            for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
                try:
                    parsed_time = datetime.strptime(current_upload_time, fmt)
                    break 
                except ValueError:
                    continue
            
            if parsed_time:
                m_item['upload_time'] = parsed_time
            else:
                # This warning is from your previous version, it's good.
                logger.warning(f"Could not parse upload_time string: '{current_upload_time}' for meeting ID {m_item.get('id')}. Setting to None.")
                m_item['upload_time'] = None
        elif not isinstance(current_upload_time, datetime): # If it's not a string AND not a datetime
            if current_upload_time is not None: # Log if it's some other unexpected type
                 logger.warning(f"Unexpected type for upload_time: {type(current_upload_time)} value: '{current_upload_time}' for meeting ID {m_item.get('id')}. Setting to None.")
            m_item['upload_time'] = None # Default to None if not already a datetime (e.g. if it was None from DB)
        
        meetings_list.append(m_item)
        
    return render_template('index.html', meetings=meetings_list)


@app.route('/meeting/<int:meeting_id>')
def meeting_detail(meeting_id):
    db = get_db()
    cursor = db.cursor()
    
    cursor.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,))
    meeting_raw = cursor.fetchone()
    if not meeting_raw:
        logger.warning(f"Attempt to access non-existent meeting_id: {meeting_id}")
        flash('Meeting not found.', 'danger')
        return redirect(url_for('index'))
    
    meeting = dict(meeting_raw)
    # Parse upload_time if it's a string (similar to index route)
    current_upload_time = meeting.get('upload_time')
    if isinstance(current_upload_time, str):
        parsed_time = None
        for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
            try:
                parsed_time = datetime.strptime(current_upload_time, fmt)
                break
            except ValueError:
                continue
        if parsed_time:
            meeting['upload_time'] = parsed_time
        else:
            logger.warning(f"Could not parse upload_time string in meeting_detail: '{current_upload_time}' for meeting ID {meeting.get('id')}. Setting to None.")
            meeting['upload_time'] = None
    elif not isinstance(current_upload_time, datetime):
        if current_upload_time is not None:
            logger.warning(f"Unexpected type for upload_time in meeting_detail: {type(current_upload_time)} value: '{current_upload_time}' for meeting ID {meeting.get('id')}. Setting to None.")
        meeting['upload_time'] = None


    cursor.execute("SELECT * FROM action_items WHERE meeting_id = ?", (meeting_id,))
    action_items_raw = cursor.fetchall()
    action_items = [dict(item) for item in action_items_raw]
    
    cursor.execute("SELECT * FROM decisions WHERE meeting_id = ?", (meeting_id,))
    decisions_raw = cursor.fetchall()
    decisions = [dict(decision) for decision in decisions_raw]
    
    logger.debug(f"Displaying details for meeting ID {meeting_id}. Summary: '{str(meeting.get('summary'))[:50]}...'")
    return render_template('meeting_detail.html', meeting=meeting, action_items=action_items, decisions=decisions)

@app.route('/action_item/<int:item_id>/toggle', methods=['POST'])
def toggle_action_item_status(item_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT status, meeting_id FROM action_items WHERE id = ?", (item_id,))
    item = cursor.fetchone()
    if not item:
        logger.warning(f"Attempt to toggle status for non-existent action_item_id: {item_id}")
        flash('Action item not found.', 'danger')
        # It might be better to redirect back to the last known good page or index
        # For now, redirecting to index if meeting_id isn't available.
        return redirect(url_for('index')) 

    new_status = 'completed' if item['status'] == 'pending' else 'pending'
    cursor.execute("UPDATE action_items SET status = ? WHERE id = ?", (new_status, item_id))
    db.commit()
    logger.info(f"Action item ID {item_id} status toggled to '{new_status}' for meeting ID {item['meeting_id']}.")
    flash(f'Action item status updated to {new_status}.', 'success')
    return redirect(url_for('meeting_detail', meeting_id=item['meeting_id']))

@app.route('/meeting/<int:meeting_id>/calendar')
def download_calendar_file(meeting_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT filename FROM meetings WHERE id = ?", (meeting_id,))
    meeting_row = cursor.fetchone()
    if not meeting_row:
        logger.warning(f"Calendar export requested for non-existent meeting_id: {meeting_id}")
        flash('Meeting not found for calendar export.', 'danger')
        return redirect(url_for('index'))

    cursor.execute("SELECT task, owner, due_date FROM action_items WHERE meeting_id = ? AND status = 'pending'", (meeting_id,))
    action_items_raw = cursor.fetchall()

    if not action_items_raw:
        logger.info(f"No pending action items to export for meeting ID {meeting_id}.")
        flash('No pending action items to export for this meeting.', 'info')
        return redirect(url_for('meeting_detail', meeting_id=meeting_id))

    cal = Calendar()
    for item_row_cal in action_items_raw:
        item_cal = dict(item_row_cal)
        event = Event()
        event.name = f"Action: {item_cal['task']}"
        description = f"Task: {item_cal['task']}"
        if item_cal['owner']:
            description += f"\nOwner: {item_cal['owner']}"
        event.description = description
        
        parsed_due_date = None
        if item_cal['due_date']:
            # Use dateparser for more robust parsing
            parsed_due_date = dateparser.parse(item_cal['due_date'], settings={'PREFER_DATES_FROM': 'future', 'RETURN_AS_TIMEZONE_AWARE': False})

        if parsed_due_date:
            event.begin = parsed_due_date 
            event.make_all_day() # Keep it simple as all-day event
        else: # No due date or unparsable
            event.begin = datetime.now().date() # Default to today, all day
            event.make_all_day()
            if item_cal['due_date']: # If there was a due_date string but it wasn't parsable
                event.description += f"\nOriginal Due Date (unparsed): {item_cal['due_date']}"
        
        cal.events.add(event)

    ics_filename = f"meeting_{meeting_id}_actions.ics"
    ics_filepath = os.path.join(app.config['UPLOAD_FOLDER'], ics_filename)
    
    try:
        with open(ics_filepath, 'w', encoding='utf-8') as f:
            f.writelines(cal.serialize_iter())
        logger.info(f"Generated ICS calendar file: {ics_filepath} for meeting ID {meeting_id} with {len(cal.events)} events.")
        
        response = send_file(ics_filepath, as_attachment=True, download_name=ics_filename, mimetype='text/calendar')
        return response
    except Exception as e:
        logger.error(f"Error generating or sending ICS file for meeting ID {meeting_id}: {e}", exc_info=True)
        flash("Error generating calendar file.", "danger")
        return redirect(url_for('meeting_detail', meeting_id=meeting_id))
    finally:
        # Ensure cleanup even if send_file fails before returning, if file was created
        if os.path.exists(ics_filepath):
            try:
                os.remove(ics_filepath)
                logger.debug(f"Removed temporary ICS file: {ics_filepath}")
            except Exception as e_rem:
                logger.error(f"Error removing temporary ICS file {ics_filepath}: {e_rem}")

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        logger.info(f"Created UPLOAD_FOLDER at {UPLOAD_FOLDER} on startup.")
    
    # The logging basicConfig might be better placed here if other modules don't set it,
    # or if you want to ensure a specific format/level for the whole app.
    # However, nlp_processor.py also calls basicConfig. The first call usually wins.
    # For more control, use Flask's app.logger or configure handlers manually.
    # For now, let's ensure the logger for 'app' has a level if not set by root.
    if not logging.getLogger().hasHandlers(): # Check if root logger has handlers
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.info("Root logger configured by app.py because no handlers were found.")
    else:
        logger.info("Root logger seems to be already configured (likely by another module).")


    logger.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5001)