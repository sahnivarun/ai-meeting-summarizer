# AI Meeting Summarizer + Action Tracker + Decision Log

An AI-powered full-stack application that transcribes meeting audio, summarizes it, extracts actionable tasks and decisions, and tracks them on an interactive dashboard with calendar integration.

## Features

* Transcribe Audio Automatically – Upload or record meeting audio and get accurate transcriptions using Whisper.
* Generate Smart Summaries – Get concise bullet-point summaries from your meeting transcripts via GPT.
* Extract Action Items & Decisions – Automatically pull out tasks with owners and due dates, plus key decisions.
* Track Everything in One Place – View and manage meetings, action items, and decisions in a unified dashboard.
* Flexible Input Options – Upload files, record live, or paste raw text.
* Calendar Integration – Download .ics files with pending action items to add to your calendar.
* Status Management – Mark tasks as pending or completed, and toggle decision statuses.
* Beautiful Calendar View – Navigate meetings by date and view their summaries directly.

## How to Run

1. Clone the Repository

git clone <https://github.com/sahnivarun/ai-meeting-summarizer>
cd ai-meeting-summarizer

2. Set Up Python Virtual Environment (Optional but Recommended)

python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate    # For Windows

3. Install Dependencies

pip install -r requirements.txt

4. Add OpenAI API Key

Create a .env file in the root directory:

OPENAI_API_KEY=your_openai_key_here

5. Run the App

python app.py

Visit http://localhost:5001 in your browser.

## NLP Components (via OpenAI)

Summary Generator: Generates 4–8 bullet point summaries from transcripts.

Action Item Extractor: Parses structured JSON (task, owner, due date).

Decision Extractor: Extracts final decisions made in meeting.

## Dashboard Modules

/ – Upload audio, paste text, or record live

/tracker – Global action item tracker (filter by owner/status)

/decision_tracker – Decision log with status toggles

/calendar – Interactive meeting calendar with day-wise detail

/meeting/<id> – Detailed view with summary, tasks, decisions, transcript

## Supported File Formats

Audio: mp3, wav, m4a, webm, flac, ogg, mp4

Text: Paste transcript in textbox

Max file size: 100MB

## Sample Flow

* Upload/record meeting or paste transcript
* Transcription → NLP summary → action items/decisions
* Results saved to DB (SQLite)
* View all meetings via dashboard/calendar
* Track pending items & update statuses
* Export .ics for calendar import

### Example Video

You can view a demo of the working project here: https://drive.google.com/file/d/1amasXuICtkUcHmBtqc1vCJg5wN3tXwyn/view?usp=drive_link

## Contributing

PRs are welcome! If you'd like to improve UI/UX, error handling, or extend NLP models (e.g., local models), feel free to fork and contribute.

## Tech Stack

* Python + Flask
* Whisper (OpenAI)
* ChatGPT (OpenAI API)
* SQLite3
* HTML + Jinja2
* JavaScript (vanilla + DOM APIs)
* CSS (custom, no external framework)

## Contact

Developer: Varun Sahni
Email: varunsahni@tamu.edu