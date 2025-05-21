# app.py
import os
import sqlite3
import logging
import openai # For specific error types like AuthenticationError

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, g, jsonify # Added jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
from ics import Calendar, Event
import dateparser

# Custom modules
from database import get_db_connection, init_db
from transcription import transcribe_audio, load_whisper_model
from nlp_processor import generate_summary, extract_action_items, extract_decisions

# Configuration (same as before)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'mp4', 'ogg', 'flac', 'webm'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

with app.app_context():
    init_db()
    logger.info("Database initialized.")
    load_whisper_model()

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

# --- HELPER FUNCTION FOR PROCESSING AUDIO (UPLOADED OR RECORDED) ---
def process_audio_file(filepath, original_filename_for_db):
    """
    Helper function to process an audio file (transcribe, NLP, DB operations).
    Returns a dictionary with status, meeting_id, summary, actions, decisions, or error message.
    """
    meeting_id = None
    summary_result = "ERROR: Processing did not complete." # Default error
    action_items_data = []
    decisions_data = []
    nlp_error_occurred = True # Assume error until proven otherwise

    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("INSERT INTO meetings (filename, processing_status) VALUES (?, ?)", 
                       (original_filename_for_db, 'uploaded'))
        meeting_id = cursor.lastrowid
        db.commit()
        logger.info(f"Meeting record created with ID: {meeting_id} for '{original_filename_for_db}', status: uploaded.")
        
        cursor.execute("UPDATE meetings SET processing_status = ? WHERE id = ?", ('transcribing', meeting_id))
        db.commit()
        logger.info(f"Status updated to 'transcribing' for meeting ID {meeting_id} ('{original_filename_for_db}').")
        transcript_text = transcribe_audio(filepath)

        if not transcript_text:
            logger.error(f"Transcription failed for meeting ID {meeting_id} ('{original_filename_for_db}').")
            summary_result = 'ERROR: Transcription failed.'
            cursor.execute("UPDATE meetings SET processing_status = ?, transcript = ?, summary = ? WHERE id = ?", 
                           ('error', 'Transcription failed.', summary_result, meeting_id))
            db.commit()
            return {'status': 'error', 'message': summary_result, 'meeting_id': meeting_id}
        
        logger.info(f"Transcription complete for meeting ID {meeting_id}. Transcript length: {len(transcript_text)} chars.")
        cursor.execute("UPDATE meetings SET transcript = ?, processing_status = ? WHERE id = ?", 
                       (transcript_text, 'processing_nlp', meeting_id))
        db.commit()
        logger.info(f"Status updated to 'processing_nlp' for meeting ID {meeting_id}.")

        logger.info(f"Starting NLP processing for meeting ID {meeting_id}...")
        summary_result = generate_summary(transcript_text) # Returns summary or "ERROR:..." string
        logger.info(f"Received summary_result (len: {len(summary_result)}): '{summary_result[:100]}...' for meeting ID {meeting_id}")
        
        action_items_data = extract_action_items(transcript_text) # Returns list or empty list if error
        logger.info(f"Received {len(action_items_data)} action items for meeting ID {meeting_id}")
        
        decisions_data = extract_decisions(transcript_text) # Returns list or empty list if error
        logger.info(f"Received {len(decisions_data)} decisions for meeting ID {meeting_id}")

        if summary_result.startswith("ERROR:"):
            logger.error(f"NLP error during summary generation for meeting ID {meeting_id}: {summary_result}")
            # nlp_error_occurred remains True
        else:
            # Check if action items or decisions also had issues, though generate_summary is the primary error flag here
            # For now, if summary is OK, we assume NLP generally worked.
            nlp_error_occurred = False 
        
        current_db_status = 'error' if nlp_error_occurred else 'completed'
        db_summary_to_store = summary_result

        logger.info(f"Updating meeting ID {meeting_id} with final summary and status '{current_db_status}'.")
        cursor.execute("UPDATE meetings SET summary = ?, processing_status = ? WHERE id = ?", 
                       (db_summary_to_store, current_db_status, meeting_id))
        
        # We always try to insert action_items_data and decisions_data
        # as they are designed to be empty lists if their specific extraction failed.
        logger.info(f"Attempting to insert {len(action_items_data)} action items for meeting ID {meeting_id}.")
        for item in action_items_data:
            cursor.execute("INSERT INTO action_items (meeting_id, task, owner, due_date) VALUES (?, ?, ?, ?)",
                           (meeting_id, item.get('task'), item.get('owner'), item.get('due_date')))
        
        logger.info(f"Attempting to insert {len(decisions_data)} decisions for meeting ID {meeting_id}.")
        for decision_text in decisions_data:
            cursor.execute("INSERT INTO decisions (meeting_id, decision_text) VALUES (?, ?)",
                           (meeting_id, decision_text))
        
        db.commit()
        logger.info(f"Database commit successful for meeting ID {meeting_id} NLP results.")
        
        return {
            'status': 'success', 
            'meeting_id': meeting_id, 
            'summary': summary_result, # Send back the summary (or error string)
            'action_items': action_items_data,
            'decisions': decisions_data,
            'nlp_error': nlp_error_occurred, # Explicit flag for overall NLP health
            'filename': original_filename_for_db
        }

    except openai.AuthenticationError as e:
        error_msg = f"ERROR: OpenAI Authentication Error (check API Key): {e}"
        logger.critical(f"Meeting ID {meeting_id if meeting_id else 'N/A'}: {error_msg}", exc_info=True)
        if meeting_id:
            db_conn_auth_err = get_db()
            cur_auth_err = db_conn_auth_err.cursor()
            cur_auth_err.execute("UPDATE meetings SET summary = ?, processing_status = ? WHERE id = ?", 
                           (error_msg, 'error', meeting_id))
            db_conn_auth_err.commit()
        return {'status': 'error', 'message': error_msg, 'meeting_id': meeting_id}
    except Exception as e:
        error_msg = f"ERROR: Unexpected error processing file {original_filename_for_db} (meeting ID {meeting_id if meeting_id else 'N/A'}): {e}"
        logger.error(error_msg, exc_info=True)
        if meeting_id:
            try:
                db_conn_gen_err = get_db()
                cur_gen_err = db_conn_gen_err.cursor()
                summary_error_text = f"Processing Error: {str(e)[:250]}" 
                cur_gen_err.execute("UPDATE meetings SET summary = ?, processing_status = ? WHERE id = ?", 
                               (summary_error_text, 'error', meeting_id))
                db_conn_gen_err.commit()
            except Exception as db_err:
                 logger.error(f"Failed to update meeting ID {meeting_id} status to error after general processing error: {db_err}")
        return {'status': 'error', 'message': f'An unexpected error occurred: {str(e)[:100]}...', 'meeting_id': meeting_id}


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST': # This is for traditional file UPLOAD
        # ... (file check logic as before) ...
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
            original_filename = secure_filename(file.filename) 
            timestamp_prefix = datetime.now().strftime("%Y%m%d%H%M%S")
            storage_filename = f"{timestamp_prefix}_{original_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], storage_filename)
            
            try:
                if not os.path.exists(app.config['UPLOAD_FOLDER']):
                    os.makedirs(app.config['UPLOAD_FOLDER'])
                file.save(filepath)
                logger.info(f"Uploaded file {original_filename} saved to {filepath} (as {storage_filename})")
                
                result = process_audio_file(filepath, original_filename) # Call helper

                if result['status'] == 'success':
                    flash_message = f'File {result["filename"]} processed.'
                    flash_category = 'success'
                    if result.get('nlp_error', False): 
                        flash_message += " However, there were issues generating some NLP results."
                        flash_category = 'warning'
                    flash(flash_message, flash_category)
                    return redirect(url_for('meeting_detail', meeting_id=result['meeting_id']))
                else:
                    # result['message'] will contain the error from process_audio_file
                    flash(f"Error processing {original_filename}: {result.get('message', 'Unknown error')}", 'danger')
                    return redirect(url_for('index'))

            except Exception as e: 
                logger.error(f"Error handling uploaded file {original_filename} before processing: {e}", exc_info=True)
                flash(f'An error occurred with file upload: {str(e)}', 'danger')
                return redirect(request.url)
        else: # File not allowed
            logger.warning(f"Upload attempt with disallowed file type: {file.filename if file else 'No file object'}")
            flash('File type not allowed', 'warning')
            return redirect(request.url)

    # --- GET request part of index route (same as before) ---
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id, filename, upload_time, processing_status, summary FROM meetings ORDER BY DATETIME(upload_time) DESC")
    meetings_raw = cursor.fetchall()
    
    meetings_list = []
    for m_raw in meetings_raw:
        m_item = dict(m_raw) 
        current_upload_time = m_item.get('upload_time')
        if isinstance(current_upload_time, str):
            parsed_time = None
            for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
                try: parsed_time = datetime.strptime(current_upload_time, fmt); break 
                except ValueError: continue
            m_item['upload_time'] = parsed_time if parsed_time else None
            if not parsed_time and current_upload_time: 
                logger.warning(f"Could not parse upload_time str: '{current_upload_time}' ID {m_item.get('id')}.")
        elif not isinstance(current_upload_time, datetime) and current_upload_time is not None:
            logger.warning(f"Unexpected type for upload_time: {type(current_upload_time)} val: '{current_upload_time}' ID {m_item.get('id')}.")
            m_item['upload_time'] = None
        meetings_list.append(m_item)
        
    return render_template('index.html', meetings=meetings_list)


@app.route('/process_recorded_audio', methods=['POST'])
def process_recorded_audio():
    logger.info("Received request to /process_recorded_audio")
    if 'audio_file' not in request.files:
        logger.error("No audio_file in request to /process_recorded_audio")
        return jsonify({'status': 'error', 'message': 'No audio data received.'}), 400
    
    file = request.files['audio_file']
    
    if file.filename == '':
        logger.error("Received audio_file with empty filename in /process_recorded_audio")
        return jsonify({'status': 'error', 'message': 'No filename for recorded audio.'}), 400

    original_filename = secure_filename(file.filename) 
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)

    try:
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        file.save(filepath)
        logger.info(f"Live recording '{original_filename}' saved to {filepath}")

        db_filename_title = f"Live Recording ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
        result = process_audio_file(filepath, db_filename_title) # Call helper

        # The 'result' dictionary now contains 'summary', 'action_items', 'decisions'
        if result['status'] == 'success':
            redirect_url = url_for('meeting_detail', meeting_id=result['meeting_id'])
            logger.info(f"Successfully processed live recording. Meeting ID: {result['meeting_id']}.")
            # Send all relevant data back to JS
            return jsonify({
                'status': 'success', 
                'meeting_id': result['meeting_id'], 
                'redirect_url': redirect_url,
                'summary': result.get('summary', "Summary not available."), # Ensure these keys exist
                'action_items': result.get('action_items', []),
                'decisions': result.get('decisions', []),
                'nlp_error': result.get('nlp_error', False) # Pass the nlp_error flag too
            })
        else:
            logger.error(f"Error processing live recording '{original_filename}': {result['message']}")
            # Send back error and potentially the meeting_id if it was created before failure
            return jsonify({'status': 'error', 'message': result.get('message', 'Unknown processing error'), 'meeting_id': result.get('meeting_id')}), 500

    except Exception as e:
        logger.error(f"Critical error handling live recording '{original_filename}': {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'Server error during recording processing: {str(e)}'}), 500


# --- Other routes (meeting_detail, toggle_action_item_status, download_calendar_file) ---
# These should be the same as the last complete version you have.
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
    current_upload_time = meeting.get('upload_time')
    if isinstance(current_upload_time, str):
        parsed_time = None
        for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
            try: parsed_time = datetime.strptime(current_upload_time, fmt); break
            except ValueError: continue
        meeting['upload_time'] = parsed_time if parsed_time else None
        if not parsed_time and current_upload_time: logger.warning(f"Could not parse upload_time str in detail: '{current_upload_time}' ID {meeting.get('id')}.")
    elif not isinstance(current_upload_time, datetime) and current_upload_time is not None:
        logger.warning(f"Unexpected type for upload_time in detail: {type(current_upload_time)} val: '{current_upload_time}' ID {meeting.get('id')}.")
        meeting['upload_time'] = None
    cursor.execute("SELECT * FROM action_items WHERE meeting_id = ?", (meeting_id,))
    action_items = [dict(item) for item in cursor.fetchall()]
    cursor.execute("SELECT * FROM decisions WHERE meeting_id = ?", (meeting_id,))
    decisions = [dict(decision) for decision in cursor.fetchall()]
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
        flash('Action item not found.', 'danger'); return redirect(url_for('index')) 
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
    if not cursor.fetchone(): 
        logger.warning(f"Calendar export for non-existent meeting_id: {meeting_id}"); flash('Meeting not found.', 'danger'); return redirect(url_for('index'))
    cursor.execute("SELECT task, owner, due_date FROM action_items WHERE meeting_id = ? AND status = 'pending'", (meeting_id,))
    action_items_raw = cursor.fetchall()
    if not action_items_raw:
        logger.info(f"No pending actions for calendar export for meeting ID {meeting_id}.")
        flash('No pending action items to export.', 'info'); return redirect(url_for('meeting_detail', meeting_id=meeting_id))
    cal = Calendar()
    for item_row_cal in action_items_raw:
        item_cal = dict(item_row_cal); event = Event(); event.name = f"Action: {item_cal['task']}"
        description = f"Task: {item_cal['task']}" + (f"\nOwner: {item_cal['owner']}" if item_cal['owner'] else "")
        event.description = description
        parsed_due_date = dateparser.parse(item_cal['due_date'], settings={'PREFER_DATES_FROM': 'future', 'RETURN_AS_TIMEZONE_AWARE': False}) if item_cal['due_date'] else None
        event.begin = parsed_due_date if parsed_due_date else datetime.now().date(); event.make_all_day()
        if not parsed_due_date and item_cal['due_date']: event.description += f"\nOriginal Due Date (unparsed): {item_cal['due_date']}"
        cal.events.add(event)
    ics_filename = f"meeting_{meeting_id}_actions.ics"; ics_filepath = os.path.join(app.config['UPLOAD_FOLDER'], ics_filename)
    try:
        with open(ics_filepath, 'w', encoding='utf-8') as f: f.writelines(cal.serialize_iter())
        logger.info(f"Generated ICS: {ics_filepath} for meeting ID {meeting_id}, {len(cal.events)} events.")
        return send_file(ics_filepath, as_attachment=True, download_name=ics_filename, mimetype='text/calendar')
    except Exception as e:
        logger.error(f"Error generating/sending ICS for meeting ID {meeting_id}: {e}", exc_info=True)
        flash("Error generating calendar file.", "danger"); return redirect(url_for('meeting_detail', meeting_id=meeting_id))
    finally:
        if os.path.exists(ics_filepath):
            try: os.remove(ics_filepath); logger.debug(f"Removed temp ICS: {ics_filepath}")
            except Exception as e_rem: logger.error(f"Error removing temp ICS {ics_filepath}: {e_rem}")

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        logger.info(f"Created UPLOAD_FOLDER at {UPLOAD_FOLDER} on startup.")
    
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.info("Root logger configured by app.py because no handlers were found.")
    else:
        logger.info("Root logger seems to be already configured (likely by another module like nlp_processor).")

    logger.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5001)