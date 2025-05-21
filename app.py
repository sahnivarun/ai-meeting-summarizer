# app.py
import os
import sqlite3
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, g
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta # Ensure datetime is imported from datetime module
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
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB upload limit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

with app.app_context():
    init_db()
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'audio_file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['audio_file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp_prefix = datetime.now().strftime("%Y%m%d%H%M%S")
            unique_filename = f"{timestamp_prefix}_{filename}"
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            meeting_id = None
            try:
                if not os.path.exists(app.config['UPLOAD_FOLDER']):
                    os.makedirs(app.config['UPLOAD_FOLDER'])
                file.save(filepath)
                logger.info(f"File {unique_filename} saved to {filepath}")

                db = get_db()
                cursor = db.cursor()
                cursor.execute("INSERT INTO meetings (filename, processing_status) VALUES (?, ?)", 
                               (unique_filename, 'uploaded'))
                meeting_id = cursor.lastrowid
                db.commit()
                
                cursor.execute("UPDATE meetings SET processing_status = ? WHERE id = ?", ('transcribing', meeting_id))
                db.commit()
                logger.info(f"Transcribing meeting ID {meeting_id} ({unique_filename})...")
                transcript_text = transcribe_audio(filepath)

                if not transcript_text:
                    logger.error(f"Transcription failed for {unique_filename}")
                    cursor.execute("UPDATE meetings SET processing_status = ?, transcript = ? WHERE id = ?", 
                                   ('error', 'Transcription failed.', meeting_id))
                    db.commit()
                    flash(f'Transcription failed for {unique_filename}. Check logs.', 'danger')
                    return redirect(url_for('index'))
                
                cursor.execute("UPDATE meetings SET transcript = ?, processing_status = ? WHERE id = ?", 
                               (transcript_text, 'processing_nlp', meeting_id))
                db.commit()
                logger.info(f"Transcription complete for meeting ID {meeting_id}. Length: {len(transcript_text)}")

                try:
                    summary = generate_summary(transcript_text)
                    action_items_data = extract_action_items(transcript_text)
                    decisions_data = extract_decisions(transcript_text)
                except ValueError as e: 
                    logger.error(f"NLP Processing Error: {e}")
                    cursor.execute("UPDATE meetings SET processing_status = ?, summary = ? WHERE id = ?", 
                                   ('error', f'NLP Error: {str(e)}', meeting_id))
                    db.commit()
                    flash(f'NLP Processing Error: {str(e)}. Please check your OpenAI API key.', 'danger')
                    return redirect(url_for('index'))

                cursor.execute("UPDATE meetings SET summary = ?, processing_status = ? WHERE id = ?", 
                               (summary, 'completed', meeting_id))
                
                for item in action_items_data:
                    cursor.execute("INSERT INTO action_items (meeting_id, task, owner, due_date) VALUES (?, ?, ?, ?)",
                                   (meeting_id, item.get('task'), item.get('owner'), item.get('due_date')))
                
                for decision_text in decisions_data:
                    cursor.execute("INSERT INTO decisions (meeting_id, decision_text) VALUES (?, ?)",
                                   (meeting_id, decision_text))
                
                db.commit()
                logger.info(f"NLP processing complete for meeting ID {meeting_id}.")
                flash(f'File {unique_filename} processed successfully!', 'success')
                return redirect(url_for('meeting_detail', meeting_id=meeting_id))

            except Exception as e:
                logger.error(f"Error processing file {unique_filename}: {e}", exc_info=True)
                if meeting_id:
                    try:
                        db_conn = get_db()
                        cur = db_conn.cursor()
                        cur.execute("UPDATE meetings SET processing_status = ? WHERE id = ?", ('error', meeting_id))
                        db_conn.commit()
                    except Exception as db_err:
                         logger.error(f"Failed to update meeting status to error: {db_err}")
                flash(f'An error occurred: {str(e)}', 'danger')
                return redirect(request.url)
        else:
            flash('File type not allowed or file error.', 'warning')
            return redirect(request.url)

    # GET request part
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id, filename, upload_time, processing_status, summary FROM meetings ORDER BY DATETIME(upload_time) DESC")
    meetings_raw = cursor.fetchall()
    
    meetings = []
    for m_raw in meetings_raw:
        m = dict(m_raw) 
        current_upload_time = m.get('upload_time')

        if isinstance(current_upload_time, str):
            parsed_time = None
            for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
                try:
                    parsed_time = datetime.strptime(current_upload_time, fmt)
                    break 
                except ValueError:
                    continue
            
            if parsed_time:
                m['upload_time'] = parsed_time
            else:
                logger.warning(f"Could not parse upload_time string: '{current_upload_time}' for meeting ID {m.get('id')}. Setting to None.")
                m['upload_time'] = None
        elif not isinstance(current_upload_time, datetime):
            if current_upload_time is not None:
                 logger.warning(f"Unexpected type for upload_time: {type(current_upload_time)} value: '{current_upload_time}' for meeting ID {m.get('id')}. Setting to None.")
            m['upload_time'] = None
        
        meetings.append(m)
        
    return render_template('index.html', meetings=meetings)


@app.route('/meeting/<int:meeting_id>')
def meeting_detail(meeting_id):
    db = get_db()
    cursor = db.cursor()
    
    cursor.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,))
    meeting_raw = cursor.fetchone()
    if not meeting_raw:
        flash('Meeting not found.', 'danger')
        return redirect(url_for('index'))
    
    meeting = dict(meeting_raw)
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
    
    return render_template('meeting_detail.html', meeting=meeting, action_items=action_items, decisions=decisions)

@app.route('/action_item/<int:item_id>/toggle', methods=['POST'])
def toggle_action_item_status(item_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT status, meeting_id FROM action_items WHERE id = ?", (item_id,))
    item = cursor.fetchone()
    if not item:
        flash('Action item not found.', 'danger')
        return redirect(url_for('index'))

    new_status = 'completed' if item['status'] == 'pending' else 'pending'
    cursor.execute("UPDATE action_items SET status = ? WHERE id = ?", (new_status, item_id))
    db.commit()
    flash(f'Action item status updated to {new_status}.', 'success')
    return redirect(url_for('meeting_detail', meeting_id=item['meeting_id']))

@app.route('/meeting/<int:meeting_id>/calendar')
def download_calendar_file(meeting_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT filename FROM meetings WHERE id = ?", (meeting_id,))
    meeting_row = cursor.fetchone() # Renamed to avoid conflict with meeting dict later
    if not meeting_row:
        flash('Meeting not found for calendar export.', 'danger')
        return redirect(url_for('index'))

    cursor.execute("SELECT task, owner, due_date FROM action_items WHERE meeting_id = ? AND status = 'pending'", (meeting_id,))
    action_items_raw = cursor.fetchall() # Renamed to avoid conflict

    if not action_items_raw:
        flash('No pending action items to export for this meeting.', 'info')
        return redirect(url_for('meeting_detail', meeting_id=meeting_id))

    cal = Calendar()
    for item_row_cal in action_items_raw: # Use a unique loop variable
        item_cal = dict(item_row_cal)
        event = Event()
        event.name = f"Action: {item_cal['task']}"
        description = f"Task: {item_cal['task']}"
        if item_cal['owner']:
            description += f"\nOwner: {item_cal['owner']}"
        event.description = description
        
        parsed_due_date = None
        if item_cal['due_date']:
            parsed_due_date = dateparser.parse(item_cal['due_date'], settings={'PREFER_DATES_FROM': 'future', 'RETURN_AS_TIMEZONE_AWARE': False})

        if parsed_due_date:
            event.begin = parsed_due_date 
            event.make_all_day()
        else:
            event.begin = datetime.now().date()
            event.make_all_day()
            if item_cal['due_date']:
                event.description += f"\nOriginal Due Date (unparsed): {item_cal['due_date']}"
        
        cal.events.add(event)

    ics_filename = f"meeting_{meeting_id}_actions.ics"
    ics_filepath = os.path.join(app.config['UPLOAD_FOLDER'], ics_filename)
    
    try:
        with open(ics_filepath, 'w', encoding='utf-8') as f: # Added encoding
            f.writelines(cal.serialize_iter())
        
        response = send_file(ics_filepath, as_attachment=True, download_name=ics_filename, mimetype='text/calendar')
        return response
    except Exception as e:
        logger.error(f"Error generating or sending ICS file: {e}", exc_info=True)
        flash("Error generating calendar file.", "danger")
        return redirect(url_for('meeting_detail', meeting_id=meeting_id))
    finally:
        if os.path.exists(ics_filepath):
            try:
                os.remove(ics_filepath)
            except Exception as e_rem:
                logger.error(f"Error removing temporary ICS file {ics_filepath}: {e_rem}")

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, host='0.0.0.0', port=5001)