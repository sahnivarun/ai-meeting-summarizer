# AI Meeting Summarizer

This project develops a system that can automatically create meeting summaries, track action items, and maintain a decision log from recorded meeting audio.

## Features

*   Upload audio files (MP3, WAV, M4A, MP4, OGG, FLAC, WEBM).
*   Automatic transcription using local OpenAI Whisper (pre-loads model on startup).
*   NLP-powered summarization, action item extraction, and decision logging using OpenAI GPT API.
*   Web-based dashboard to view processed meetings and their details.
*   Mark action items as pending/completed.
*   Download pending action items as an `.ics` calendar file (uses `dateparser` for better due date recognition).
*   Basic loading indicator during file processing.

## Project Structure