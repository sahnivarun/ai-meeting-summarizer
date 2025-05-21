# nlp_processor.py
import openai
import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not openai.api_key:
    logger.warning("OPENAI_API_KEY not found in .env file. NLP processing will fail.")

def get_llm_response(prompt, system_message="You are a helpful assistant.", model="gpt-3.5-turbo"):
    if not openai.api_key:
        raise ValueError("OpenAI API key not configured.")
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return None

def generate_summary(transcript):
    prompt = f"""
    Please provide a concise summary of the following meeting transcript.
    Focus on the key topics discussed and main outcomes.

    Transcript:
    ---
    {transcript}
    ---

    Summary:
    """
    system_message = "You are an expert meeting summarizer."
    return get_llm_response(prompt, system_message)

def extract_action_items(transcript):
    prompt = f"""
    Analyze the following meeting transcript and extract all action items.
    For each action item, identify the task, the assigned owner (if mentioned), and any due date (if mentioned).
    Present the output as a JSON list of objects. Each object should have 'task', 'owner', and 'due_date' keys.
    If an owner or due date is not mentioned, use null or an empty string for that field.

    Example for one action item:
    {{
        "task": "Prepare the quarterly report",
        "owner": "Alice",
        "due_date": "Next Friday"
    }}

    If no action items are found, return an empty list [].

    Transcript:
    ---
    {transcript}
    ---

    JSON Output:
    """
    system_message = "You are an intelligent assistant skilled at extracting structured information like action items from text."
    response_text = get_llm_response(prompt, system_message)
    if response_text:
        try:
            # The LLM might sometimes add explanations around the JSON. Try to extract just the JSON part.
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text and response_text.strip().startswith('['): # Sometimes just ``` with no json label
                 response_text = response_text.split('```')[1].split('```')[0].strip()
            
            # Ensure it starts and ends with list brackets if not empty
            if response_text and not response_text.startswith('['):
                response_text = '[' + response_text # Be careful with this logic, depends on LLM output
            if response_text and not response_text.endswith(']'):
                response_text = response_text + ']'

            # Basic cleanup
            response_text = response_text.replace('\\n', ' ').strip()

            # Validate if it's empty list string '[]' or contains items
            if response_text == "[]":
                return []
            
            # Attempt to parse
            items = json.loads(response_text)
            
            # Ensure it's a list
            if not isinstance(items, list):
                logger.warning(f"LLM did not return a list for action items. Response: {response_text}")
                return [] # Or handle as an error / retry

            # Normalize keys and provide defaults
            normalized_items = []
            for item in items:
                if isinstance(item, dict): # Ensure item is a dictionary
                    normalized_items.append({
                        'task': item.get('task', 'N/A'),
                        'owner': item.get('owner'), # Can be None
                        'due_date': item.get('due_date') # Can be None
                    })
                else:
                    logger.warning(f"Skipping non-dict item in action items: {item}")
            return normalized_items
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for action items: {e}. Response: {response_text}")
            return [] # Return empty list on parsing error
        except Exception as e:
            logger.error(f"Unexpected error processing action items: {e}. Response: {response_text}")
            return []
    return []


def extract_decisions(transcript):
    prompt = f"""
    Analyze the following meeting transcript and extract key decisions made.
    Present the output as a JSON list of strings, where each string is a decision.
    If no decisions are found, return an empty list [].

    Example:
    [
        "The team will adopt the new software by Q3.",
        "Project Alpha budget is approved."
    ]

    Transcript:
    ---
    {transcript}
    ---

    JSON Output:
    """
    system_message = "You are an intelligent assistant skilled at extracting structured information like decisions from text."
    response_text = get_llm_response(prompt, system_message)
    if response_text:
        try:
            # The LLM might sometimes add explanations around the JSON. Try to extract just the JSON part.
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text and response_text.strip().startswith('['):
                 response_text = response_text.split('```')[1].split('```')[0].strip()

            # Basic cleanup
            response_text = response_text.replace('\\n', ' ').strip()

            if response_text == "[]":
                return []
            
            decisions = json.loads(response_text)
            if not isinstance(decisions, list):
                 logger.warning(f"LLM did not return a list for decisions. Response: {response_text}")
                 return []
            return [str(d) for d in decisions if isinstance(d, str)] # Ensure items are strings
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for decisions: {e}. Response: {response_text}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error processing decisions: {e}. Response: {response_text}")
            return []
    return []

if __name__ == '__main__':
    # Example usage (requires OPENAI_API_KEY in .env)
    sample_transcript = """
    Alice: Good morning, everyone. Let's start with project Phoenix. Bob, any updates?
    Bob: Yes, Alice. We've completed the initial design phase. We need to decide on the cloud provider by end of this week. I suggest AWS.
    Charlie: I agree with AWS. Also, Alice, can you schedule a follow-up meeting for next Tuesday?
    Alice: Okay, decision made, we'll go with AWS for Project Phoenix. Bob, please prepare a cost analysis. Charlie, I'll send out the invite.
    Bob: Action item for me: prepare cost analysis for AWS. Due by Friday.
    Alice: And an action item for me is to schedule the meeting for next Tuesday.
    Charlie: Great.
    """
    if not openai.api_key:
        print("Skipping NLP tests as OPENAI_API_KEY is not set.")
    else:
        print("--- Generating Summary ---")
        summary = generate_summary(sample_transcript)
        print(summary)

        print("\n--- Extracting Action Items ---")
        actions = extract_action_items(sample_transcript)
        print(json.dumps(actions, indent=2))

        print("\n--- Extracting Decisions ---")
        decisions = extract_decisions(sample_transcript)
        print(json.dumps(decisions, indent=2))
