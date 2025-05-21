# app.py
import os
import sqlite3
import logging
import openai 
import json # For passing data to calendar JS

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, g, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
from ics import Calendar, Event
import dateparser

# Custom modules
from database import get_db_connection, init_db # init_db will now handle new decision columns
from transcription import transcribe_audio, load_whisper_model
from nlp_processor import generate_summary, extract_action_items, extract_decisions

# Configuration
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
    init_db() # This will now attempt to add new columns to 'decisions' if they don't exist
    logger.info("Database initialized/verified.")
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

# --- HELPER FUNCTION FOR PROCESSING AUDIO ---
def process_audio_file(filepath, actual_stored_filename):
    meeting_id = None; summary_result = "ERROR: Processing did not complete." 
    action_items_data = []; decisions_data = []; nlp_error_occurred = True 
    try:
        db = get_db(); cursor = db.cursor()
        # Store the actual filename that is on disk
        cursor.execute("INSERT INTO meetings (filename, processing_status) VALUES (?, ?)", 
                       (actual_stored_filename, 'uploaded'))
        meeting_id = cursor.lastrowid; db.commit()
        logger.info(f"Meeting record created ID: {meeting_id} for stored file '{actual_stored_filename}'.")
        
        cursor.execute("UPDATE meetings SET processing_status = ? WHERE id = ?", ('transcribing', meeting_id)); db.commit()
        logger.info(f"Status to 'transcribing' for ID {meeting_id}.")
        transcript_text = transcribe_audio(filepath)

        if not transcript_text:
            logger.error(f"Transcription failed for ID {meeting_id}."); summary_result = 'ERROR: Transcription failed.'
            cursor.execute("UPDATE meetings SET processing_status = ?, transcript = ?, summary = ? WHERE id = ?", 
                           ('error', 'Transcription failed.', summary_result, meeting_id)); db.commit()
            return {'status': 'error', 'message': summary_result, 'meeting_id': meeting_id, 'summary':summary_result, 'action_items':[], 'decisions':[]}
        
        logger.info(f"Transcription complete for ID {meeting_id}. Length: {len(transcript_text)} chars.")
        cursor.execute("UPDATE meetings SET transcript = ?, processing_status = ? WHERE id = ?", 
                       (transcript_text, 'processing_nlp', meeting_id)); db.commit()
        logger.info(f"Status to 'processing_nlp' for ID {meeting_id}.")

        logger.info(f"Starting NLP for ID {meeting_id}..."); summary_result = generate_summary(transcript_text)
        logger.info(f"Summary result for ID {meeting_id} (len: {len(summary_result)}): '{summary_result[:100]}...'")
        action_items_data = extract_action_items(transcript_text); logger.info(f"{len(action_items_data)} AIs for ID {meeting_id}")
        decisions_data = extract_decisions(transcript_text); logger.info(f"{len(decisions_data)} decisions for ID {meeting_id}")
        
        nlp_error_occurred = summary_result.startswith("ERROR:")
        if nlp_error_occurred: logger.error(f"NLP error for ID {meeting_id}: {summary_result}")
        
        current_db_status = 'error' if nlp_error_occurred else 'completed'; db_summary_to_store = summary_result
        logger.info(f"Updating ID {meeting_id} with summary and status '{current_db_status}'.")
        cursor.execute("UPDATE meetings SET summary = ?, processing_status = ? WHERE id = ?", 
                       (db_summary_to_store, current_db_status, meeting_id))
        
        logger.info(f"Inserting {len(action_items_data)} AIs and {len(decisions_data)} decisions for ID {meeting_id}.")
        for item in action_items_data: 
            cursor.execute("INSERT INTO action_items (meeting_id, task, owner, due_date) VALUES (?, ?, ?, ?)", 
                           (meeting_id, item.get('task'), item.get('owner'), item.get('due_date')))
        for decision_text in decisions_data: 
            # Decisions now have a status, default is 'open' from DB schema
            cursor.execute("INSERT INTO decisions (meeting_id, decision_text) VALUES (?, ?)", 
                           (meeting_id, decision_text))
        db.commit(); logger.info(f"DB commit for ID {meeting_id} NLP results.")
        
        return {'status': 'success', 'meeting_id': meeting_id, 'summary': summary_result, 
                'action_items': action_items_data, 'decisions': decisions_data, 
                'nlp_error': nlp_error_occurred, 'filename': actual_stored_filename}
    except openai.AuthenticationError as e:
        error_msg = f"ERROR: OpenAI Auth Error: {e}"; logger.critical(f"ID {meeting_id or 'N/A'}: {error_msg}", exc_info=True)
        if meeting_id:
            try: db=get_db();cur=db.cursor();cur.execute("UPDATE meetings SET summary = ?, processing_status = ? WHERE id = ?", (error_msg, 'error', meeting_id));db.commit()
            except Exception as db_e: logger.error(f"DB error on auth fail for {meeting_id}: {db_e}")
        return {'status': 'error', 'message': error_msg, 'meeting_id': meeting_id, 'summary':error_msg, 'action_items':[], 'decisions':[]}
    except Exception as e:
        error_msg = f"ERROR: Unexpected error for {actual_stored_filename} (ID {meeting_id or 'N/A'}): {e}"; logger.error(error_msg, exc_info=True)
        if meeting_id:
            try: db=get_db();cur=db.cursor();cur.execute("UPDATE meetings SET summary = ?, processing_status = ? WHERE id = ?", (f"Proc. Error: {str(e)[:250]}", 'error', meeting_id));db.commit()
            except Exception as db_err: logger.error(f"DB error on general fail for {meeting_id}: {db_err}")
        return {'status': 'error', 'message': f'Unexpected error: {str(e)[:100]}...', 'meeting_id': meeting_id, 'summary':f"Proc. Error: {str(e)[:250]}", 'action_items':[], 'decisions':[]}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST': 
        if 'audio_file' not in request.files: logger.warning("Upload: No file part."); flash('No file part', 'danger'); return redirect(request.url)
        file = request.files['audio_file']
        if file.filename == '': logger.warning("Upload: No selected file."); flash('No selected file', 'danger'); return redirect(request.url)
        if file and allowed_file(file.filename):
            original_filename_display = secure_filename(file.filename) # For display
            timestamp_prefix = datetime.now().strftime('%Y%m%d%H%M%S')
            # This storage_filename is what gets stored in DB and on disk
            storage_filename = f"{timestamp_prefix}_{original_filename_display}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], storage_filename)
            try:
                if not os.path.exists(app.config['UPLOAD_FOLDER']): os.makedirs(app.config['UPLOAD_FOLDER'])
                file.save(filepath); logger.info(f"Uploaded '{original_filename_display}' to {filepath}")
                # Pass the actual storage_filename to be stored in the DB's filename column
                result = process_audio_file(filepath, storage_filename) 
                if result['status'] == 'success':
                    flash_msg = f'File "{result["filename"]}" processed.' + (" Issues with NLP." if result.get('nlp_error') else "")
                    flash(flash_msg, 'warning' if result.get('nlp_error') else 'success')
                    return redirect(url_for('meeting_detail', meeting_id=result['meeting_id']))
                else:
                    flash(f"Error processing {original_filename_display}: {result.get('message', 'Unknown error')}", 'danger')
                    return redirect(url_for('index'))
            except Exception as e: logger.error(f"Error handling {original_filename_display}: {e}", exc_info=True); flash(f'Error: {str(e)}', 'danger'); return redirect(request.url)
        else: logger.warning(f"Upload: Disallowed type: {file.filename if file else 'N/A'}"); flash('File type not allowed', 'warning'); return redirect(request.url)
    
    db = get_db(); cursor = db.cursor()
    cursor.execute("SELECT id, filename, upload_time, processing_status, summary FROM meetings ORDER BY DATETIME(upload_time) DESC")
    meetings_raw = cursor.fetchall()
    meetings_list = []
    for m_raw in meetings_raw:
        m_item = dict(m_raw); current_upload_time = m_item.get('upload_time')
        if isinstance(current_upload_time, str): # Should be datetime if DB converters work
            parsed_time = None
            for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
                try: parsed_time = datetime.strptime(current_upload_time, fmt); break 
                except ValueError: continue
            m_item['upload_time'] = parsed_time
            if not parsed_time and current_upload_time: logger.warning(f"Could not parse upload_time str: '{current_upload_time}' ID {m_item.get('id')}.")
        elif not isinstance(current_upload_time, datetime) and current_upload_time is not None:
            logger.warning(f"Unexpected type for upload_time: {type(current_upload_time)} val: '{current_upload_time}' ID {m_item.get('id')}.")
            m_item['upload_time'] = None # Should not happen if DB converters are set
        meetings_list.append(m_item)
    return render_template('index.html', meetings=meetings_list)

@app.route('/process_recorded_audio', methods=['POST'])
def process_recorded_audio():
    logger.info("Request to /process_recorded_audio")
    if 'audio_file' not in request.files: logger.error("/process_recorded_audio: No audio_file."); return jsonify({'status': 'error', 'message': 'No audio data.'}), 400
    file = request.files['audio_file']
    if file.filename == '': logger.error("/process_recorded_audio: Empty filename."); return jsonify({'status': 'error', 'message': 'No filename.'}), 400
    
    # JS sends a filename like "live_recording_YYYYMMDDTHHMMSS.webm"
    # This is the actual filename to be stored on disk and in DB
    actual_stored_filename = secure_filename(file.filename) 
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], actual_stored_filename)
    try:
        if not os.path.exists(app.config['UPLOAD_FOLDER']): os.makedirs(app.config['UPLOAD_FOLDER'])
        file.save(filepath); logger.info(f"Live recording '{actual_stored_filename}' saved to {filepath}")
        
        # Pass the actual_stored_filename to be stored in the DB's filename column
        result = process_audio_file(filepath, actual_stored_filename) 
        
        if result['status'] == 'success':
            redirect_url = url_for('meeting_detail', meeting_id=result['meeting_id'])
            logger.info(f"Processed live recording. Meeting ID: {result['meeting_id']}.")
            return jsonify({
                'status': 'success', 'meeting_id': result['meeting_id'], 'redirect_url': redirect_url,
                'summary': result.get('summary', "N/A"), 
                'action_items': result.get('action_items', []),
                'decisions': result.get('decisions', []), # Decisions from NLP, not yet with status from DB
                'nlp_error': result.get('nlp_error', False)
            })
        else:
            logger.error(f"Error processing live recording '{actual_stored_filename}': {result.get('message')}")
            return jsonify({'status': 'error', 'message': result.get('message', 'Unknown error'), 'meeting_id': result.get('meeting_id')}), 500
    except Exception as e:
        logger.error(f"Critical error handling live recording '{actual_stored_filename}': {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'Server error: {str(e)}'}), 500

@app.route('/tracker')
def action_tracker():
    db = get_db(); cursor = db.cursor()
    query = "SELECT ai.id, ai.task, ai.owner, ai.due_date, ai.status, ai.meeting_id, m.filename as meeting_filename, m.upload_time as meeting_upload_time FROM action_items ai JOIN meetings m ON ai.meeting_id = m.id ORDER BY DATETIME(m.upload_time) DESC, CASE ai.status WHEN 'pending' THEN 1 ELSE 2 END, ai.due_date ASC NULLS LAST;"
    cursor.execute(query); action_items_raw = cursor.fetchall()
    all_action_items = []
    for item_raw in action_items_raw:
        item = dict(item_raw); current_meeting_time = item.get('meeting_upload_time')
        if isinstance(current_meeting_time, str):
            parsed_time = None
            for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
                try: parsed_time = datetime.strptime(current_meeting_time, fmt); break
                except ValueError: continue
            item['meeting_upload_time'] = parsed_time
            if not parsed_time and current_meeting_time: logger.warning(f"Tracker: Could not parse meeting_upload_time str: '{current_meeting_time}' for AI ID {item.get('id')}.")
        elif not isinstance(current_meeting_time, datetime) and current_meeting_time is not None: # Should be datetime
            logger.warning(f"Tracker: Unexpected type for meeting_upload_time: {type(current_meeting_time)} for AI ID {item.get('id')}.")
            item['meeting_upload_time'] = None
        all_action_items.append(item)
    logger.info(f"Fetched {len(all_action_items)} action items for tracker.")
    return render_template('tracker.html', all_action_items=all_action_items)

@app.route('/decision_tracker')
def decision_tracker():
    db = get_db(); cursor = db.cursor()
    query = """
    SELECT d.id, d.decision_text, d.status, d.resolution_notes, d.meeting_id,
           m.filename as meeting_filename, m.upload_time as meeting_upload_time
    FROM decisions d JOIN meetings m ON d.meeting_id = m.id
    ORDER BY DATETIME(m.upload_time) DESC, d.id DESC;
    """
    cursor.execute(query); decisions_raw = cursor.fetchall()
    all_decisions = []
    for item_raw in decisions_raw:
        item = dict(item_raw); current_meeting_time = item.get('meeting_upload_time')
        if isinstance(current_meeting_time, str):
            parsed_time = None
            for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
                try: parsed_time = datetime.strptime(current_meeting_time, fmt); break
                except ValueError: continue
            item['meeting_upload_time'] = parsed_time
            if not parsed_time and current_meeting_time: logger.warning(f"DecisionTracker: Could not parse meeting_upload_time: '{current_meeting_time}' for Decision ID {item.get('id')}.")
        elif not isinstance(current_meeting_time, datetime) and current_meeting_time is not None:
            logger.warning(f"DecisionTracker: Unexpected type for meeting_upload_time: {type(current_meeting_time)} for Decision ID {item.get('id')}.")
            item['meeting_upload_time'] = None
        all_decisions.append(item)
    logger.info(f"Fetched {len(all_decisions)} decisions for decision log.")
    return render_template('decision_tracker.html', all_decisions=all_decisions)

# --- NEW API ENDPOINT FOR CALENDAR DETAILS ---
@app.route('/api/meeting_details/<int:meeting_id>')
def api_meeting_details(meeting_id):
    db = get_db()
    cursor = db.cursor()
    
    meeting_data = {}
    # Fetch meeting
    cursor.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,))
    meeting_raw = cursor.fetchone()
    if not meeting_raw:
        return jsonify({"error": "Meeting not found"}), 404
    meeting_data['meeting'] = dict(meeting_raw)
    # Ensure datetime is ISO format string for JSON
    if isinstance(meeting_data['meeting'].get('upload_time'), datetime):
        meeting_data['meeting']['upload_time'] = meeting_data['meeting']['upload_time'].isoformat()


    # Fetch action items
    cursor.execute("SELECT * FROM action_items WHERE meeting_id = ?", (meeting_id,))
    meeting_data['action_items'] = [dict(row) for row in cursor.fetchall()]

    # Fetch decisions (including new status field)
    cursor.execute("SELECT id, decision_text, status, resolution_notes FROM decisions WHERE meeting_id = ?", (meeting_id,))
    meeting_data['decisions'] = [dict(row) for row in cursor.fetchall()]
    
    logger.debug(f"API: Sending details for meeting ID {meeting_id}")
    return jsonify(meeting_data)

@app.route('/calendar')
def calendar_view():
    db = get_db(); cursor = db.cursor()
    cursor.execute("SELECT id, filename, upload_time FROM meetings WHERE processing_status = 'completed' OR (processing_status = 'error' AND transcript IS NOT NULL AND transcript != 'Transcription failed.') ORDER BY upload_time")
    meetings = cursor.fetchall()
    meetings_by_date = {}
    for meeting_row in meetings:
        meeting = dict(meeting_row); upload_time = meeting.get('upload_time')
        if isinstance(upload_time, str): # Should be datetime due to DB converter
            parsed_time = None
            for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
                try: parsed_time = datetime.strptime(upload_time, fmt); break
                except ValueError: continue
            upload_time = parsed_time
        if isinstance(upload_time, datetime):
            date_str = upload_time.strftime('%Y-%m-%d')
            if date_str not in meetings_by_date: meetings_by_date[date_str] = []
            meetings_by_date[date_str].append({'id': meeting['id'], 'filename': meeting['filename'], 'upload_time': upload_time.isoformat()})
        else: logger.warning(f"Calendar: Meeting ID {meeting['id']} has invalid upload_time: {meeting.get('upload_time')}")
    meetings_by_date_json = json.dumps(meetings_by_date)
    logger.info(f"Prepared {len(meetings_by_date)} dates for calendar.")
    return render_template('calendar_view.html', meetings_by_date_json=meetings_by_date_json)

@app.route('/meeting/<int:meeting_id>')
def meeting_detail(meeting_id):
    db = get_db(); cursor = db.cursor()
    cursor.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,))
    meeting_raw = cursor.fetchone()
    if not meeting_raw: logger.warning(f"Detail: Non-existent meeting_id: {meeting_id}"); flash('Meeting not found.', 'danger'); return redirect(url_for('index'))
    meeting = dict(meeting_raw); current_upload_time = meeting.get('upload_time')
    if isinstance(current_upload_time, str): # Should be datetime
        parsed_time = None
        for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
            try: parsed_time = datetime.strptime(current_upload_time, fmt); break
            except ValueError: continue
        meeting['upload_time'] = parsed_time
        if not parsed_time and current_upload_time: logger.warning(f"Detail: Could not parse upload_time: '{current_upload_time}' ID {meeting.get('id')}.")
    elif not isinstance(current_upload_time, datetime) and current_upload_time is not None:
        logger.warning(f"Detail: Unexpected type for upload_time: {type(current_upload_time)} ID {meeting.get('id')}.")
        meeting['upload_time'] = None # Should not happen
    cursor.execute("SELECT * FROM action_items WHERE meeting_id = ?", (meeting_id,))
    action_items = [dict(item) for item in cursor.fetchall()]
    # Fetch decisions including new status field
    cursor.execute("SELECT id, decision_text, status, resolution_notes FROM decisions WHERE meeting_id = ?", (meeting_id,))
    decisions = [dict(decision) for decision in cursor.fetchall()]
    logger.debug(f"Displaying details for meeting ID {meeting_id}.")
    return render_template('meeting_detail.html', meeting=meeting, action_items=action_items, decisions=decisions)

@app.route('/action_item/<int:item_id>/toggle', methods=['POST'])
def toggle_action_item_status(item_id):
    db = get_db(); cursor = db.cursor()
    cursor.execute("SELECT status, meeting_id FROM action_items WHERE id = ?", (item_id,))
    item = cursor.fetchone()
    if not item: logger.warning(f"ToggleAI: Non-existent ID: {item_id}"); flash('Action item not found.', 'danger'); return redirect(request.referrer or url_for('index'))
    new_status = 'completed' if item['status'] == 'pending' else 'pending'
    cursor.execute("UPDATE action_items SET status = ? WHERE id = ?", (new_status, item_id)); db.commit()
    logger.info(f"AI ID {item_id} status toggled to '{new_status}'.")
    flash(f'Action item status updated to {new_status}.', 'success')
    next_url = request.args.get('next')
    if next_url: logger.debug(f"Redirecting to 'next' URL: {next_url}"); return redirect(next_url)
    return redirect(url_for('meeting_detail', meeting_id=item['meeting_id']))

# --- NEW ROUTE FOR TOGGLING DECISION STATUS ---
@app.route('/decision/<int:decision_id>/toggle_status', methods=['POST'])
def toggle_decision_status(decision_id):
    db = get_db()
    cursor = db.cursor()
    
    # Determine the new status: 
    # Could get it from form, or implement a specific cycle
    # The form in decision_tracker.html and meeting_detail.html sends 'new_status'
    # For the Re-Open from implemented state in meeting_detail, it might send new_status_direct
    new_status_from_form = request.form.get('new_status')
    new_status_direct = request.args.get('new_status_direct') # For GET-like param on POST action

    if new_status_direct: # If explicitly passed in URL param for specific buttons
        new_status = new_status_direct
    elif new_status_from_form:
        new_status = new_status_from_form
    else: # Default simple toggle if no specific new_status is provided (should not happen with current forms)
        cursor.execute("SELECT status FROM decisions WHERE id = ?", (decision_id,))
        current_decision = cursor.fetchone()
        if not current_decision:
            flash('Decision not found.', 'danger')
            return redirect(request.referrer or url_for('index'))
        new_status = 'implemented' if current_decision['status'] == 'open' else 'open'
        
    # You might want to add logic for resolution_notes here if adding a form for it
    # For now, just updating status
    cursor.execute("UPDATE decisions SET status = ? WHERE id = ?", (new_status, decision_id))
    db.commit()
    logger.info(f"Decision ID {decision_id} status updated to '{new_status}'.")
    flash(f'Decision status updated to {new_status}.', 'success')

    next_url = request.args.get('next')
    if next_url:
        return redirect(next_url)
    # Fallback if no 'next' URL (should ideally always have one from the forms)
    # To find the meeting_id, we need to query it if not passed
    cursor.execute("SELECT meeting_id FROM decisions WHERE id = ?", (decision_id,))
    decision_info = cursor.fetchone()
    if decision_info:
        return redirect(url_for('meeting_detail', meeting_id=decision_info['meeting_id']))
    return redirect(url_for('decision_tracker')) # Ultimate fallback


@app.route('/meeting/<int:meeting_id>/delete', methods=['POST'])
def delete_meeting(meeting_id):
    logger.info(f"Attempting to delete meeting ID: {meeting_id}")
    db = get_db(); cursor = db.cursor()
    cursor.execute("SELECT filename FROM meetings WHERE id = ?", (meeting_id,))
    meeting_record = cursor.fetchone()
    if not meeting_record: logger.warning(f"Delete: Meeting ID {meeting_id} not found."); flash('Meeting not found.', 'danger'); return redirect(url_for('index'))
    actual_filename_to_delete = meeting_record['filename'] 
    try:
        # ON DELETE CASCADE in DB schema should handle action_items and decisions
        cursor.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,)); 
        logger.info(f"Deleted meeting record {meeting_id} and via CASCADE its AIs/Decisions.")
        db.commit()
        file_to_delete_path = os.path.join(app.config['UPLOAD_FOLDER'], actual_filename_to_delete)
        if os.path.exists(file_to_delete_path):
            os.remove(file_to_delete_path); logger.info(f"Deleted audio file: {file_to_delete_path}")
        else: logger.warning(f"Audio file for deletion not found: {file_to_delete_path}")
        flash(f'Meeting "{actual_filename_to_delete}" deleted.', 'success')
    except sqlite3.Error as e:
        db.rollback(); logger.error(f"DB error deleting meeting {meeting_id}: {e}", exc_info=True); flash(f'DB error: {e}', 'danger')
    except OSError as e: logger.error(f"OS error deleting file for meeting {meeting_id}: {e}", exc_info=True); flash(f'File deletion error: {e}', 'warning') # DB part might have succeeded
    except Exception as e:
        db.rollback(); logger.error(f"General error deleting meeting {meeting_id}: {e}", exc_info=True); flash(f'Error deleting: {e}', 'danger')
    return redirect(url_for('index'))

@app.route('/meeting/<int:meeting_id>/calendar') # .ics download
def download_calendar_file(meeting_id):
    # ... (download_calendar_file route remains the same) ...
    db = get_db(); cursor = db.cursor()
    cursor.execute("SELECT filename FROM meetings WHERE id = ?", (meeting_id,))
    if not cursor.fetchone(): logger.warning(f"CalendarDL: Non-existent meeting_id: {meeting_id}"); flash('Meeting not found.', 'danger'); return redirect(url_for('index'))
    cursor.execute("SELECT task, owner, due_date FROM action_items WHERE meeting_id = ? AND status = 'pending'", (meeting_id,))
    action_items_raw = cursor.fetchall()
    if not action_items_raw: logger.info(f"CalendarDL: No pending actions for meeting ID {meeting_id}."); flash('No pending action items.', 'info'); return redirect(url_for('meeting_detail', meeting_id=meeting_id))
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
    except Exception as e: logger.error(f"Error generating/sending ICS for meeting ID {meeting_id}: {e}", exc_info=True); flash("Error generating calendar file.", "danger"); return redirect(url_for('meeting_detail', meeting_id=meeting_id))
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
        logger.info("Root logger configured by app.py (main block).")
    else:
        logger.info("Root logger previously configured.")
    logger.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5001)