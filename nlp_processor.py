# nlp_processor.py
import openai
import os
import json
import logging
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.critical("CRITICAL: OPENAI_API_KEY not found in .env file. NLP processing WILL FAIL.")
    # In a real app, you might exit or disable NLP features here.
    # For now, we'll let it try and fail, but the log is critical.
    client = None
else:
    logger.info("OpenAI API Key found. Initializing OpenAI client.")
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        # You could make a simple test call here if desired, e.g., list models
        # client.models.list() 
        logger.info("OpenAI client initialized successfully.")
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to initialize OpenAI client: {e}", exc_info=True)
        client = None


def get_llm_response(prompt_details: str, user_prompt: str, system_message: str = "You are a helpful assistant.", model: str = "gpt-3.5-turbo"):
    """
    Sends a prompt to the OpenAI API and returns the response content or an error message.

    Args:
        prompt_details (str): A short description of what this prompt is for (e.g., "Summary Generation").
        user_prompt (str): The actual prompt to send to the LLM.
        system_message (str): The system message for the LLM.
        model (str): The OpenAI model to use.

    Returns:
        str: The LLM's response content, or a string starting with "ERROR:" if an issue occurred.
    """
    if not client:
        error_msg = "ERROR: OpenAI client not initialized. Check API key and initial setup."
        logger.error(f"{prompt_details}: {error_msg}")
        return error_msg

    logger.info(f"Sending API request for: {prompt_details}. Model: {model}. Prompt length: {len(user_prompt)} chars.")
    # To be very verbose for debugging (remove in production if too noisy):
    # logger.debug(f"Full prompt for {prompt_details}:\nSYSTEM: {system_message}\nUSER: {user_prompt[:500]}...") # Log first 500 chars

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = response.choices[0].message.content
        if content:
            logger.info(f"{prompt_details}: API call successful. Response length: {len(content)} chars.")
            return content.strip()
        else:
            logger.warning(f"{prompt_details}: API call successful but content is empty.")
            return "ERROR: API returned empty content."
            
    except openai.APIConnectionError as e:
        error_msg = f"ERROR: OpenAI API Connection Error: {e}"
        logger.error(f"{prompt_details}: {error_msg}", exc_info=True)
        return error_msg
    except openai.RateLimitError as e:
        error_msg = f"ERROR: OpenAI API Rate Limit Exceeded: {e}"
        logger.error(f"{prompt_details}: {error_msg}", exc_info=True)
        return error_msg
    except openai.AuthenticationError as e:
        error_msg = f"ERROR: OpenAI API Authentication Error (check API Key): {e}"
        logger.error(f"{prompt_details}: {error_msg}", exc_info=True)
        return error_msg
    except openai.APIError as e: # Catch other OpenAI API errors
        error_msg = f"ERROR: OpenAI API Error: {e}"
        logger.error(f"{prompt_details}: {error_msg}", exc_info=True)
        return error_msg
    except Exception as e: # Catch any other unexpected errors
        error_msg = f"ERROR: An unexpected error occurred during API call for {prompt_details}: {e}"
        logger.error(f"{prompt_details}: {error_msg}", exc_info=True)
        return error_msg

def generate_summary(transcript: str) -> str:
    if not transcript or transcript.isspace():
        logger.warning("generate_summary called with empty or whitespace-only transcript.")
        return "ERROR: Transcript is empty, cannot generate summary."

    prompt = f"""
    You are an expert meeting summarizer. Analyze the following transcript and generate a clean, concise summary.
    Focus on the key topics discussed, decisions made, and outcomes agreed upon.

    Avoid irrelevant small talk or greetings. Present the summary in 4–8 bullet points if possible.
    This summary will be used in a dashboard for team tracking, so be objective and informative.

    Transcript:
    ---
    {transcript}
    ---

    Summary:
    """

    system_message = "You are an expert meeting summarizer."
    logger.info("Requesting summary generation from LLM.")
    summary_response = get_llm_response("Summary Generation", prompt, system_message)
    
    if summary_response.startswith("ERROR:"):
        logger.error(f"Failed to generate summary: {summary_response}")
        return summary_response # Return the error message as the summary
    
    logger.info("Summary generated successfully.")
    return summary_response

def extract_action_items(transcript: str) -> list:
    if not transcript or transcript.isspace():
        logger.warning("extract_action_items called with empty or whitespace-only transcript.")
        return [] # Return empty list if transcript is empty

    prompt = f"""
    You are a smart assistant extracting action items from a meeting transcript.
    Identify **all** action items clearly discussed or implied. For each item, return a JSON object with:
    - `task`: the task or follow-up action
    - `owner`: person responsible (null or "" if not mentioned)
    - `due_date`: deadline or timeline (null or "N/A" if not mentioned)

    Respond ONLY in a valid JSON list of objects like this:

    [
    {{
        "task": "Prepare the quarterly report",
        "owner": "Alice",
        "due_date": "Next Friday"
    }},
    ...
    ]

    If no action items are present, return an empty list `[]`.

    Transcript:
    ---
    {transcript}
    ---

    JSON Output:
    """

    system_message = "You are an intelligent assistant skilled at extracting structured information like action items from text."
    logger.info("Requesting action item extraction from LLM.")
    response_text = get_llm_response("Action Item Extraction", prompt, system_message)
    
    if response_text.startswith("ERROR:"):
        logger.error(f"Action item extraction failed: {response_text}")
        return [] # Return empty list on API error

    parsed_items = []
    try:
        logger.debug(f"Attempting to parse action items JSON. Raw response snippet: {response_text[:200]}")
        if '```json' in response_text:
            json_str = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text and (response_text.strip().startswith('[') or response_text.strip().startswith('{')):
            json_str = response_text.split('```')[1].split('```')[0].strip()
        else:
            json_str = response_text

        items = json.loads(json_str)
        
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    parsed_items.append({
                        'task': item.get('task', 'N/A'),
                        'owner': item.get('owner'),
                        'due_date': item.get('due_date')
                    })
                else:
                    logger.warning(f"Skipping non-dict item in action items: {item}")
            logger.info(f"Successfully parsed {len(parsed_items)} action items.")
        else:
            logger.warning(f"LLM did not return a list for action items, but a {type(items)}. Response: {response_text[:200]}")
            
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON for action items: {e}. Response snippet: {response_text[:500]}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error processing action items after API call: {e}. Response snippet: {response_text[:500]}", exc_info=True)
            
    return parsed_items


def extract_decisions(transcript: str) -> list:
    if not transcript or transcript.isspace():
        logger.warning("extract_decisions called with empty or whitespace-only transcript.")
        return []

    prompt = f"""
    You are a structured-thinking assistant. From this meeting transcript, extract **all decisions** made by the participants.

    Format your output as a JSON array of plain English strings, each summarizing one decision.
    Do not include tasks or suggestions unless they were explicitly agreed as decisions.

    Examples:
    [
    "The team will migrate to the new CRM system in Q2.",
    "Budget for Project Alpha has been approved."
    ]

    If no decisions are found, return `[]`.

    Transcript:
    ---
    {transcript}
    ---

    JSON Output:
    """

    system_message = "You are an intelligent assistant skilled at extracting structured information like decisions from text."
    logger.info("Requesting decision extraction from LLM.")
    response_text = get_llm_response("Decision Extraction", prompt, system_message)

    if response_text.startswith("ERROR:"):
        logger.error(f"Decision extraction failed: {response_text}")
        return [] # Return empty list on API error

    parsed_decisions = []
    try:
        logger.debug(f"Attempting to parse decisions JSON. Raw response snippet: {response_text[:200]}")
        if '```json' in response_text:
            json_str = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text and (response_text.strip().startswith('[') or response_text.strip().startswith('{')):
            json_str = response_text.split('```')[1].split('```')[0].strip()
        else:
            json_str = response_text

        decisions = json.loads(json_str)

        if isinstance(decisions, list):
            parsed_decisions = [str(d) for d in decisions if isinstance(d, (str, int, float))]
            logger.info(f"Successfully parsed {len(parsed_decisions)} decisions.")
        else:
            logger.warning(f"LLM did not return a list for decisions, but a {type(decisions)}. Response: {response_text[:200]}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON for decisions: {e}. Response snippet: {response_text[:500]}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error processing decisions after API call: {e}. Response snippet: {response_text[:500]}", exc_info=True)
            
    return parsed_decisions


if __name__ == '__main__':
    # This block is for direct testing of nlp_processor.py
    # Ensure your .env file has OPENAI_API_KEY set.
    
    # You can paste a short sample transcript here for testing
    sample_transcript_for_test = """
    Alice: Good morning. Let's review the Q2 results. The numbers are looking good overall.
    Bob: Yes, the marketing campaign seems to have paid off. We saw a 20% increase in leads.
    Charlie: However, the conversion rate from lead to sale is still a bit low. We need to address that.
    Alice: Agreed. Decision: We will focus on improving the sales funnel for Q3.
    Bob: Action item for me: Analyze the current sales funnel and identify bottlenecks by next Friday.
    Charlie: Action item for me: Research CRM tools that could help automate follow-ups, present findings in two weeks.
    Alice: Excellent. Any other business? No? Meeting adjourned.
    """
    # sample_transcript_for_test = "" # Test with empty transcript
    # sample_transcript_for_test = "Just a short test." # Test with very short transcript


    if not OPENAI_API_KEY or not client:
        print("CRITICAL: OpenAI API Key not found or client not initialized. Skipping direct NLP tests.")
    else:
        print(f"\n--- DIRECT TEST: Generating Summary from sample ---\nTranscript length: {len(sample_transcript_for_test)} chars")
        summary = generate_summary(sample_transcript_for_test)
        print(f"\nSUMMARY OUTPUT:\n{summary}")

        print(f"\n--- DIRECT TEST: Extracting Action Items from sample ---")
        actions = extract_action_items(sample_transcript_for_test)
        print(f"\nACTION ITEMS OUTPUT:\n{json.dumps(actions, indent=2)}")

        print(f"\n--- DIRECT TEST: Extracting Decisions from sample ---")
        decisions = extract_decisions(sample_transcript_for_test)
        print(f"\nDECISIONS OUTPUT:\n{json.dumps(decisions, indent=2)}")