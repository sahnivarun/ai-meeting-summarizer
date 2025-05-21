# app.py
import os
import sqlite3
import logging
import openai 
import json 

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, g, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
from ics import Calendar, Event
import dateparser

# Custom modules
from database import get_db_connection, init_db 
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
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

with app.app_context():
    init_db() 
    logger.info("Database initialized/verified by app.py.")
    load_whisper_model()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None: db.close()

def get_db():
    db = getattr(g, '_database', None)
    if db is None: db = g._database = get_db_connection()
    return db

# --- MODIFIED HELPER FUNCTION FOR AUDIO PROCESSING ---
def process_audio_file(filepath, actual_stored_filename, user_provided_title=None, original_uploaded_filename_for_default_title=None):
    meeting_id = None; summary_result = "ERROR: Initial processing error." 
    action_items_data = []; decisions_from_nlp = []; nlp_error_occurred = True 
    
    current_time_for_title = datetime.now()
    current_dt_str = current_time_for_title.strftime('%Y-%m-%d %H:%M')

    if user_provided_title and user_provided_title.strip():
        final_meeting_title = user_provided_title.strip()
        logger.info(f"User provided title: '{final_meeting_title}'")
    else:
        if "live_recording" in actual_stored_filename.lower():
            final_meeting_title = f"Live Recording ({current_dt_str})"
        elif original_uploaded_filename_for_default_title:
            final_meeting_title = f"{original_uploaded_filename_for_default_title} ({current_dt_str})"
        else: 
            final_meeting_title = f"Uploaded File ({current_dt_str})"
        logger.info(f"No user title for audio, generated default: '{final_meeting_title}' (based on file: '{actual_stored_filename}')")

    try:
        db = get_db(); cursor = db.cursor()
        cursor.execute("""
            INSERT INTO meetings (filename, processing_status, upload_time, meeting_title) 
            VALUES (?, ?, ?, ?)
            """, (actual_stored_filename, 'uploaded', current_time_for_title, final_meeting_title))
        meeting_id = cursor.lastrowid; db.commit()
        logger.info(f"PROCESSED: Meeting record created ID: {meeting_id} for file '{actual_stored_filename}' with DB title '{final_meeting_title}'.")
        
        cursor.execute("UPDATE meetings SET processing_status = ? WHERE id = ?", ('transcribing', meeting_id)); db.commit()
        transcript_text = transcribe_audio(filepath)

        if not transcript_text:
            summary_result = 'ERROR: Transcription failed.'
            cursor.execute("UPDATE meetings SET processing_status = ?, transcript = ?, summary = ? WHERE id = ?", 
                           ('error', 'Transcription failed.', summary_result, meeting_id)); db.commit() # meeting_title already set
            logger.error(f"PROCESSED: Transcription failed for ID {meeting_id}.")
            return {'status': 'error', 'message': summary_result, 'meeting_id': meeting_id, 'summary':summary_result, 'action_items':[], 'decisions':[], 'meeting_title': final_meeting_title}
        
        cursor.execute("UPDATE meetings SET transcript = ?, processing_status = ? WHERE id = ?", 
                       (transcript_text, 'processing_nlp', meeting_id)); db.commit()
        logger.info(f"PROCESSED: Transcription OK for ID {meeting_id}. Length: {len(transcript_text)}. Status to 'processing_nlp'.")

        summary_result = generate_summary(transcript_text)
        action_items_data = extract_action_items(transcript_text)
        decisions_from_nlp = extract_decisions(transcript_text) 
        nlp_error_occurred = summary_result.startswith("ERROR:")
        
        current_db_status = 'error' if nlp_error_occurred else 'completed'
               
        # The final_meeting_title is now either user-provided or the "Mode (timestamp)" default.
        cursor.execute("UPDATE meetings SET summary = ?, processing_status = ?, meeting_title = ? WHERE id = ?", 
                       (summary_result, current_db_status, final_meeting_title, meeting_id))
        
        for item in action_items_data: 
            cursor.execute("INSERT INTO action_items (meeting_id, task, owner, due_date) VALUES (?, ?, ?, ?)", (meeting_id, item.get('task'), item.get('owner'), item.get('due_date')))
        for decision_text in decisions_from_nlp: 
            cursor.execute("INSERT INTO decisions (meeting_id, decision_text) VALUES (?, ?)", (meeting_id, decision_text)) 
        db.commit()
        
        cursor.execute("SELECT id, decision_text, status, resolution_notes FROM decisions WHERE meeting_id = ?", (meeting_id,))
        processed_decisions_for_return = [dict(row) for row in cursor.fetchall()]
        logger.info(f"PROCESSED: NLP stage for ID {meeting_id} finished. Error: {nlp_error_occurred}. Summary: '{summary_result[:50]}...'")
        return {'status': 'success', 'meeting_id': meeting_id, 'summary': summary_result, 
                'action_items': action_items_data, 'decisions': processed_decisions_for_return, 
                'nlp_error': nlp_error_occurred, 'filename': actual_stored_filename, 'meeting_title': final_meeting_title}
    except openai.AuthenticationError as e:
        error_msg = f"ERROR: OpenAI Auth Error: {e}"; logger.critical(f"PROCESSED (ID {meeting_id or 'N/A'}): {error_msg}", exc_info=False)
        if meeting_id:
            try: db=get_db();cur=db.cursor();cur.execute("UPDATE meetings SET summary = ?, processing_status = ? WHERE id = ?", (error_msg, 'error', meeting_id));db.commit()
            except: pass 
        return {'status': 'error', 'message': error_msg, 'meeting_id': meeting_id, 'summary':error_msg, 'action_items':[], 'decisions':[], 'meeting_title': final_meeting_title}
    except Exception as e:
        error_msg = f"ERROR: Unexpected error processing {actual_stored_filename} (ID {meeting_id or 'N/A'}): {e}"; logger.error(error_msg, exc_info=True)
        if meeting_id:
            try: db=get_db();cur=db.cursor();cur.execute("UPDATE meetings SET summary = ?, processing_status = ? WHERE id = ?", (f"Proc. Error: {str(e)[:250]}", 'error', meeting_id));db.commit()
            except: pass 
        return {'status': 'error', 'message': f'Unexpected error: {str(e)[:100]}...', 'meeting_id': meeting_id, 'summary':f"Proc. Error: {str(e)[:250]}", 'action_items':[], 'decisions':[], 'meeting_title': final_meeting_title}

# --- MODIFIED HELPER FUNCTION FOR TEXT TRANSCRIPT PROCESSING ---
def process_text_input(transcript_text, user_provided_title=None):
    meeting_id = None; summary_result = "ERROR: Initial processing error." 
    action_items_data = []; decisions_from_nlp = []; nlp_error_occurred = True
    current_time_for_title = datetime.now()
    current_dt_str = current_time_for_title.strftime('%Y-%m-%d %H:%M')

    if user_provided_title and user_provided_title.strip():
        final_meeting_title = user_provided_title.strip()
        logger.info(f"User provided title for text input: '{final_meeting_title}'")
    else:
        final_meeting_title = f"Meeting Transcription ({current_dt_str})"
        logger.info(f"No user title for text input, generated default: '{final_meeting_title}'")
    
    placeholder_filename = f"text_input_{current_time_for_title.strftime('%Y%m%d%H%M%S')}.txt"

    try:
        db = get_db(); cursor = db.cursor()
        cursor.execute("""
            INSERT INTO meetings (filename, transcript, processing_status, upload_time, meeting_title) 
            VALUES (?, ?, ?, ?, ?)
            """, (placeholder_filename, transcript_text, 'processing_nlp', current_time_for_title, final_meeting_title))
        meeting_id = cursor.lastrowid; db.commit()
        logger.info(f"TEXT_PROC: Meeting record ID: {meeting_id}, Title: '{final_meeting_title}', Placeholder Filename: '{placeholder_filename}'. Status 'processing_nlp'.")

        summary_result = generate_summary(transcript_text)
        action_items_data = extract_action_items(transcript_text)
        decisions_from_nlp = extract_decisions(transcript_text) 
        nlp_error_occurred = summary_result.startswith("ERROR:")
        current_db_status = 'error' if nlp_error_occurred else 'completed'
        
        cursor.execute("UPDATE meetings SET summary = ?, processing_status = ?, meeting_title = ? WHERE id = ?", 
                       (summary_result, current_db_status, final_meeting_title, meeting_id)) # Use the already set final_meeting_title
        
        for item in action_items_data: 
            cursor.execute("INSERT INTO action_items (meeting_id, task, owner, due_date) VALUES (?, ?, ?, ?)", (meeting_id, item.get('task'), item.get('owner'), item.get('due_date')))
        for decision_text in decisions_from_nlp: 
            cursor.execute("INSERT INTO decisions (meeting_id, decision_text) VALUES (?, ?)", (meeting_id, decision_text)) 
        db.commit()
        
        cursor.execute("SELECT id, decision_text, status, resolution_notes FROM decisions WHERE meeting_id = ?", (meeting_id,))
        processed_decisions_for_return = [dict(row) for row in cursor.fetchall()]
        logger.info(f"TEXT_PROC: NLP stage for ID {meeting_id} finished. Error: {nlp_error_occurred}.")
        return {'status': 'success', 'meeting_id': meeting_id, 'summary': summary_result, 'action_items': action_items_data, 'decisions': processed_decisions_for_return, 'nlp_error': nlp_error_occurred, 'filename': placeholder_filename, 'meeting_title': final_meeting_title}
    # ... (rest of process_text_input's except blocks - same as your provided version) ...
    except openai.AuthenticationError as e:
        error_msg = f"ERROR: OpenAI Auth Error: {e}"; logger.critical(f"TEXT_PROC (ID {meeting_id or 'N/A'}): {error_msg}", exc_info=False)
        if meeting_id:
            try: db_err_conn=get_db();cur_err=db_err_conn.cursor();cur_err.execute("UPDATE meetings SET summary = ?, processing_status = ? WHERE id = ?", (error_msg, 'error', meeting_id));db_err_conn.commit()
            except Exception as db_e_auth: logger.error(f"DB error on auth fail for {meeting_id}: {db_e_auth}")
        return {'status': 'error', 'message': error_msg, 'meeting_id': meeting_id, 'summary':error_msg, 'action_items':[], 'decisions':[], 'meeting_title': final_meeting_title}
    except Exception as e:
        error_msg = f"ERROR: Unexpected error processing text input (ID {meeting_id or 'N/A'}): {e}"; logger.error(error_msg, exc_info=True)
        if meeting_id:
            try: db_err_conn=get_db();cur_err=db_err_conn.cursor();cur_err.execute("UPDATE meetings SET summary = ?, processing_status = ? WHERE id = ?", (f"Proc. Error: {str(e)[:250]}", 'error', meeting_id));db_err_conn.commit()
            except Exception as db_e_gen: logger.error(f"DB error on general fail for {meeting_id}: {db_e_gen}")
        return {'status': 'error', 'message': f'Unexpected error: {str(e)[:100]}...', 'meeting_id': meeting_id, 'summary':f"Proc. Error: {str(e)[:250]}", 'action_items':[], 'decisions':[], 'meeting_title': final_meeting_title}


# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    # ... (This route is the same as your last complete version, which correctly passes original_uploaded_filename) ...
    if request.method == 'POST': 
        user_meeting_title_upload = request.form.get('meeting_title_upload', '').strip()
        logger.info(f"Upload: Received meeting_title_upload: '{user_meeting_title_upload}'")
        if 'audio_file' not in request.files: flash('No audio file part in request.', 'danger'); return redirect(request.url)
        file = request.files['audio_file']
        if file.filename == '': flash('No audio file selected.', 'danger'); return redirect(request.url)
        if file and allowed_file(file.filename):
            original_uploaded_filename = file.filename 
            secured_basename_tuple = os.path.splitext(secure_filename(original_uploaded_filename)) # secure_filename on base only
            storage_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{secured_basename_tuple[0]}{secured_basename_tuple[1]}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], storage_filename)
            try:
                if not os.path.exists(app.config['UPLOAD_FOLDER']): os.makedirs(app.config['UPLOAD_FOLDER'])
                file.save(filepath); logger.info(f"Uploaded '{original_uploaded_filename}' to {filepath} (stored as {storage_filename})")
                result = process_audio_file(filepath, storage_filename, user_meeting_title_upload, original_uploaded_filename_for_default_title=original_uploaded_filename) 
                if result['status'] == 'success':
                    display_title = result.get('meeting_title', result.get("filename", "Meeting"))
                    flash_msg = f'Meeting "{display_title}" processed.' + (" Issues with NLP." if result.get('nlp_error') else "")
                    flash(flash_msg, 'warning' if result.get('nlp_error') else 'success')
                    return redirect(url_for('meeting_detail', meeting_id=result['meeting_id']))
                else:
                    flash(f"Error processing {original_uploaded_filename}: {result.get('message', 'Unknown error')}", 'danger')
                    return redirect(url_for('index'))
            except Exception as e: 
                logger.error(f"Error handling upload of {original_uploaded_filename}: {e}", exc_info=True)
                flash(f'Upload Error: {str(e)}', 'danger'); return redirect(request.url)
        else: 
            flash('File type not allowed or invalid file.', 'warning'); return redirect(request.url)
    db = get_db(); cursor = db.cursor()
    cursor.execute("SELECT id, filename, upload_time, processing_status, summary, meeting_title FROM meetings ORDER BY upload_time DESC, id DESC")
    meetings_raw = cursor.fetchall()
    meetings_list = []
    for m_raw in meetings_raw:
        m_item = dict(m_raw); display_time = m_item.get('upload_time') 
        if not isinstance(display_time, datetime) and display_time is not None: 
            parsed_time = None 
            if isinstance(display_time, str):
                for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
                    try: parsed_time = datetime.strptime(display_time, fmt); break 
                    except ValueError: continue
            m_item['display_time_for_list'] = parsed_time
            if not parsed_time and display_time : logger.warning(f"Index: display_time '{display_time}' for ID {m_item.get('id')} parse issue.")
        else: m_item['display_time_for_list'] = display_time
        m_item['display_title_for_list'] = m_item.get('meeting_title') or m_item.get('filename') or "Untitled Meeting"
        meetings_list.append(m_item)
    return render_template('index.html', meetings=meetings_list)

@app.route('/process_recorded_audio', methods=['POST'])
def process_recorded_audio():
    # ... (This route is the same as your last complete version, passes user_meeting_title_record) ...
    user_meeting_title_record = request.form.get('meeting_title_record', '').strip()
    logger.info(f"Record: Received meeting_title_record: '{user_meeting_title_record}'")
    if 'audio_file' not in request.files: return jsonify({'status': 'error', 'message': 'No audio data.'}), 400
    file = request.files['audio_file']
    if file.filename == '': return jsonify({'status': 'error', 'message': 'No filename.'}), 400
    actual_stored_filename = secure_filename(file.filename) 
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], actual_stored_filename)
    try:
        if not os.path.exists(app.config['UPLOAD_FOLDER']): os.makedirs(app.config['UPLOAD_FOLDER'])
        file.save(filepath); logger.info(f"Live recording '{actual_stored_filename}' saved to {filepath}")
        result = process_audio_file(filepath, actual_stored_filename, user_meeting_title_record, original_uploaded_filename_for_default_title=actual_stored_filename) 
        if result['status'] == 'success':
            return jsonify({'status': 'success', 'meeting_id': result['meeting_id'], 'redirect_url': url_for('meeting_detail', meeting_id=result['meeting_id']), 'summary': result.get('summary', "N/A"), 'action_items': result.get('action_items', []), 'decisions': result.get('decisions', []), 'nlp_error': result.get('nlp_error', False), 'meeting_title': result.get('meeting_title', actual_stored_filename) })
        else: return jsonify({'status': 'error', 'message': result.get('message'), 'meeting_id': result.get('meeting_id')}), 500
    except Exception as e: logger.error(f"Crit err handling live rec '{actual_stored_filename}': {e}", exc_info=True); return jsonify({'status': 'error', 'message': f'Server error: {str(e)}'}), 500

@app.route('/process_text_transcript', methods=['POST'])
def process_text_transcript():
    # ... (This route is the same as your last complete version) ...
    user_meeting_title = request.form.get('meeting_title_text', '').strip()
    transcript_text = request.form.get('transcript_text', '').strip()
    if not transcript_text: flash('Transcript text cannot be empty.', 'danger'); return redirect(url_for('index'))
    result = process_text_input(transcript_text, user_meeting_title) 
    if result['status'] == 'success':
        display_title = result.get('meeting_title', "Text Meeting")
        flash_msg = f'Meeting "{display_title}" (from text) processed.' + (" Issues with NLP." if result.get('nlp_error') else "")
        flash(flash_msg, 'warning' if result.get('nlp_error') else 'success')
        return redirect(url_for('meeting_detail', meeting_id=result['meeting_id']))
    else: flash(f"Error processing text transcript: {result.get('message', 'Unknown error')}", 'danger'); return redirect(url_for('index'))

# ... (All other routes: /tracker, /decision_tracker, /api/meeting_details, /calendar, /meeting/<id>, toggles, delete, .ics - remain IDENTICAL to your provided version)
@app.route('/tracker') 
def action_tracker():
    db = get_db(); cursor = db.cursor()
    query = "SELECT ai.id, ai.task, ai.owner, ai.due_date, ai.status, ai.meeting_id, COALESCE(m.meeting_title, m.filename) as meeting_filename, m.upload_time as meeting_upload_time FROM action_items ai JOIN meetings m ON ai.meeting_id = m.id ORDER BY DATETIME(m.upload_time) DESC, CASE ai.status WHEN 'pending' THEN 1 ELSE 2 END, ai.due_date ASC NULLS LAST;"
    cursor.execute(query); items_raw = cursor.fetchall(); all_items = []
    for r_raw in items_raw: 
        item = dict(r_raw); mt = item.get('meeting_upload_time')
        if not isinstance(mt,datetime) and mt is not None: 
            pt = None; 
            if isinstance(mt, str):
                for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S'):
                    try: pt = datetime.strptime(mt, fmt); break 
                    except ValueError: continue
            item['meeting_upload_time'] = pt
            if not pt and mt : logger.warning(f"Tracker: time parse fail: '{mt}' AI ID {item.get('id')}.")
        all_items.append(item)
    return render_template('tracker.html', all_action_items=all_items)

@app.route('/decision_tracker')
def decision_tracker():
    db = get_db(); cursor = db.cursor()
    query = "SELECT d.id, d.decision_text, d.status, d.resolution_notes, d.meeting_id, COALESCE(m.meeting_title, m.filename) as meeting_filename, m.upload_time as meeting_upload_time FROM decisions d JOIN meetings m ON d.meeting_id = m.id ORDER BY DATETIME(m.upload_time) DESC, d.id DESC;"
    cursor.execute(query); items_raw = cursor.fetchall(); all_items = []
    for r_raw in items_raw:
        item = dict(r_raw); mt = item.get('meeting_upload_time')
        if not isinstance(mt,datetime) and mt is not None: 
            pt = None; 
            if isinstance(mt, str):
                for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S'):
                    try: pt = datetime.strptime(mt, fmt); break 
                    except ValueError: continue
            item['meeting_upload_time'] = pt
            if not pt and mt: logger.warning(f"DecisionTracker: time parse fail: '{mt}' Dec ID {item.get('id')}.")
        all_items.append(item)
    return render_template('decision_tracker.html', all_decisions=all_items)

@app.route('/api/meeting_details/<int:meeting_id>')
def api_meeting_details(meeting_id):
    db = get_db(); cursor = db.cursor()
    m_data = {}; cursor.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)); m_raw = cursor.fetchone()
    if not m_raw: return jsonify({"error": "Meeting not found"}), 404
    m = dict(m_raw)
    for k in ['upload_time']: 
        val = m.get(k)
        if isinstance(val, datetime): m[k] = val.isoformat()
        elif val is not None: m[k] = str(val) 
    m_data['meeting'] = m
    cursor.execute("SELECT * FROM action_items WHERE meeting_id = ?", (meeting_id,)); m_data['action_items'] = [dict(r) for r in cursor.fetchall()]
    cursor.execute("SELECT id, decision_text, status, resolution_notes FROM decisions WHERE meeting_id = ?", (meeting_id,)); m_data['decisions'] = [dict(r) for r in cursor.fetchall()]
    return jsonify(m_data)

@app.route('/calendar')
def calendar_view():
    db = get_db(); cursor = db.cursor()
    cursor.execute("SELECT id, filename, upload_time, processing_status, meeting_title FROM meetings ORDER BY upload_time DESC")
    meetings_raw = cursor.fetchall(); meetings_by_date = {}
    for m_row in meetings_raw:
        m = dict(m_row); event_primary_time = m.get('upload_time')
        if not isinstance(event_primary_time, datetime) and event_primary_time is not None:
            parsed_time=None
            if isinstance(event_primary_time, str):
                for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S'): 
                    try: parsed_time = datetime.strptime(event_primary_time, fmt); break
                    except ValueError: continue
            event_primary_time = parsed_time
        if isinstance(event_primary_time, datetime):
            date_str = event_primary_time.strftime('%Y-%m-%d')
            if date_str not in meetings_by_date: meetings_by_date[date_str] = []
            display_title_for_calendar = m.get('meeting_title') or m.get('filename') or "Untitled Event" 
            meeting_entry = {'id': m['id'], 'display_title': display_title_for_calendar, 'processing_status': m.get('processing_status'), 'event_time_iso': event_primary_time.isoformat() }
            if m.get('upload_time') and isinstance(m['upload_time'], datetime): meeting_entry['upload_time_iso'] = m['upload_time'].isoformat()
            meetings_by_date[date_str].append(meeting_entry)
        else: logger.warning(f"Calendar: Meeting ID {m['id']} invalid event_time: {event_primary_time}")
    return render_template('calendar_view.html', meetings_by_date_json=json.dumps(meetings_by_date))

@app.route('/meeting/<int:meeting_id>')
def meeting_detail(meeting_id):
    db = get_db(); cursor = db.cursor()
    cursor.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,))
    m_raw = cursor.fetchone()
    if not m_raw: flash('Meeting not found.', 'danger'); return redirect(url_for('index'))
    m = dict(m_raw); k = 'upload_time'; val = m.get(k)
    if not isinstance(val, datetime) and val is not None:
        pt=None; 
        if isinstance(val, str):
            for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S'):
                try: pt = datetime.strptime(val, fmt); break
                except ValueError: continue
        m[k] = pt
        if not pt and val: logger.warning(f"Detail: Could not parse {k}: '{val}' ID {m.get('id')}.")
    action_items = [dict(r) for r in cursor.execute("SELECT * FROM action_items WHERE meeting_id = ?", (meeting_id,)).fetchall()]
    decisions = [dict(r) for r in cursor.execute("SELECT id, decision_text, status, resolution_notes FROM decisions WHERE meeting_id = ?", (meeting_id,)).fetchall()]
    return render_template('meeting_detail.html', meeting=m, action_items=action_items, decisions=decisions)

@app.route('/action_item/<int:item_id>/toggle', methods=['POST'])
def toggle_action_item_status(item_id):
    db=get_db();cur=db.cursor();cur.execute("SELECT status,meeting_id FROM action_items WHERE id=?",(item_id,));item=cur.fetchone()
    if not item:flash('Action item not found.','danger');return redirect(request.referrer or url_for('index'))
    new_s='completed' if item['status']=='pending' else 'pending';cur.execute("UPDATE action_items SET status=? WHERE id=?",(new_s,item_id));db.commit()
    flash(f'Action Item status updated to {new_s}.','success');next_url=request.args.get('next');return redirect(next_url or url_for('meeting_detail',meeting_id=item['meeting_id']))

@app.route('/decision/<int:decision_id>/toggle_status', methods=['POST'])
def toggle_decision_status(decision_id):
    db=get_db();cur=db.cursor();new_s_form=request.form.get('new_status');new_s_direct=request.args.get('new_status_direct')
    new_s = 'open'; 
    if new_s_direct:new_s=new_s_direct
    elif new_s_form:new_s=new_s_form 
    else:
        cur.execute("SELECT status FROM decisions WHERE id=?",(decision_id,));curr_d=cur.fetchone()
        if not curr_d: flash('Decision not found.', 'danger'); return redirect(request.referrer or url_for('decision_tracker'))
        new_s='implemented' if curr_d['status']=='open' else 'open'
    cur.execute("UPDATE decisions SET status=? WHERE id=?",(new_s,decision_id));db.commit();flash(f'Decision status updated to {new_s}.','success')
    next_url=request.args.get('next');
    if next_url:return redirect(next_url)
    cur.execute("SELECT meeting_id FROM decisions WHERE id=?",(decision_id,));d_info=cur.fetchone()
    return redirect(url_for('meeting_detail',meeting_id=d_info['meeting_id']) if d_info else url_for('decision_tracker'))

@app.route('/meeting/<int:meeting_id>/delete', methods=['POST'])
def delete_meeting(meeting_id):
    db=get_db();cur=db.cursor();cur.execute("SELECT filename,meeting_title FROM meetings WHERE id=?",(meeting_id,));m_rec=cur.fetchone()
    if not m_rec:flash('Meeting not found.','danger');return redirect(url_for('index'))
    disk_filename=m_rec['filename'];display_title=m_rec['meeting_title'] or disk_filename
    try:
        cur.execute("DELETE FROM action_items WHERE meeting_id = ?", (meeting_id,)) 
        cur.execute("DELETE FROM decisions WHERE meeting_id = ?", (meeting_id,))  
        cur.execute("DELETE FROM meetings WHERE id=?",(meeting_id,));db.commit();logger.info(f"Deleted meeting ID {meeting_id} data.")
        if disk_filename : 
            f_path=os.path.join(app.config['UPLOAD_FOLDER'],disk_filename)
            if os.path.exists(f_path):os.remove(f_path);logger.info(f"Deleted file: {f_path}")
            else:logger.warning(f"File for deletion not found: {f_path} (Title: '{display_title}')")
        else: logger.info(f"No disk file associated for meeting: {display_title}")
        flash(f'Meeting data for "{display_title}" deleted.','success')
    except Exception as e:db.rollback();logger.error(f"Error deleting meeting {meeting_id}:{e}",exc_info=True);flash(f'Error deleting: {str(e)}','danger')
    return redirect(url_for('index'))

@app.route('/meeting/<int:meeting_id>/calendar') 
def download_calendar_file(meeting_id):
    db=get_db();cur=db.cursor()
    cur.execute("SELECT filename FROM meetings WHERE id=?",(meeting_id,))
    if not cur.fetchone():flash('Meeting not found for .ics export.','danger');return redirect(url_for('index'))
    cur.execute("SELECT ai.task,ai.owner,ai.due_date FROM action_items ai JOIN meetings m ON ai.meeting_id=m.id WHERE ai.meeting_id=? AND ai.status='pending'",(meeting_id,));items_raw=cur.fetchall()
    if not items_raw:flash('No actionable items for this processed meeting to export.','info');return redirect(url_for('meeting_detail',meeting_id=meeting_id))
    cal=Calendar()
    for item_row_cal in items_raw:
        item_cal=dict(item_row_cal);event=Event();event.name=f"Action: {item_cal['task']}";description=f"Task: {item_cal['task']}"+(f"\nOwner: {item_cal['owner']}" if item_cal['owner'] else "");event.description=description
        parsed_due_date = dateparser.parse(item_cal['due_date'], settings={'PREFER_DATES_FROM': 'future', 'RETURN_AS_TIMEZONE_AWARE': False}) if item_cal['due_date'] else None
        event.begin = parsed_due_date if parsed_due_date else datetime.now().date();event.make_all_day()
        if not parsed_due_date and item_cal['due_date']: event.description += f"\nOriginal Due Date (unparsed): {item_cal['due_date']}"
        cal.events.add(event)
    ics_filename = f"meeting_{meeting_id}_actions.ics"; ics_filepath = os.path.join(app.config['UPLOAD_FOLDER'], ics_filename)
    try:
        with open(ics_filepath, 'w', encoding='utf-8') as f: f.writelines(cal.serialize_iter())
        logger.info(f"Generated ICS: {ics_filepath} for meeting ID {meeting_id}, {len(cal.events)} events.")
        return send_file(ics_filepath, as_attachment=True, download_name=ics_filename, mimetype='text/calendar')
    except Exception as e: logger.error(f"Error generating/sending ICS for meeting ID {meeting_id}: {e}", exc_info=True);flash("Error generating calendar file.", "danger");return redirect(url_for('meeting_detail',meeting_id=meeting_id))
    finally:
        if os.path.exists(ics_filepath):
            try: os.remove(ics_filepath); logger.debug(f"Removed temp ICS: {ics_filepath}")
            except Exception as e_rem: logger.error(f"Error removing temp ICS {ics_filepath}: {e_rem}")

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
    if not logging.getLogger().hasHandlers(): 
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5001)