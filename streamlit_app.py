import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
from datetime import datetime
import logging
import re
import io # For handling PDF data in memory
import httpx # Re-import httpx

# Import ReportLab components
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.units import inch
from reportlab.lib import colors

# --- Page Config (MUST be the first Streamlit command) ---
st.set_page_config(page_title="Bulk Company Research", layout="wide")

# Configure logging (optional for Streamlit, but can be helpful)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration & API Client Setup ---
load_dotenv() # Load .env file for local development

PERPLEXITY_API_KEY = None
SOURCE_MESSAGE = "Key Source: Not found yet."
API_KEY_LOADED_SUCCESSFULLY = False

# 1. Try environment variable
_raw_key_env = os.getenv('PERPLEXITY_API_KEY')
if _raw_key_env:
    PERPLEXITY_API_KEY = _raw_key_env.strip()
    if PERPLEXITY_API_KEY: # Check if non-empty after strip
        SOURCE_MESSAGE = "Key Source: Environment Variable"
        API_KEY_LOADED_SUCCESSFULLY = True
        logging.info(f"{SOURCE_MESSAGE}")
    else:
        logging.warning("Found PERPLEXITY_API_KEY in env vars, but it was empty.")

# 2. If not found in env, try st.secrets
if not API_KEY_LOADED_SUCCESSFULLY:
    logging.info("API key not found in env vars, trying st.secrets...")
    SOURCE_MESSAGE = "Key Source: Streamlit Secrets"
    try:
        _raw_key_secrets = st.secrets.get("PERPLEXITY_API_KEY") # Use .get for safety
        if _raw_key_secrets:
            PERPLEXITY_API_KEY = _raw_key_secrets.strip()
            if PERPLEXITY_API_KEY:
                logging.info("Loaded API key from st.secrets")
                API_KEY_LOADED_SUCCESSFULLY = True
                # SOURCE_MESSAGE already set
            else:
                logging.warning("Found PERPLEXITY_API_KEY in st.secrets, but it was empty after stripping.")
                SOURCE_MESSAGE = "Key Source: Streamlit Secrets (Empty Key!)"
        else:
             logging.warning("PERPLEXITY_API_KEY not found in st.secrets.")
             SOURCE_MESSAGE = "Key Source: Streamlit Secrets (Not Found)"
             
    except FileNotFoundError:
        logging.info("secrets.toml file not found (expected locally). Skipping st.secrets.")
        SOURCE_MESSAGE = "Key Source: Streamlit Secrets (File Not Found)"
    except Exception as e:
        logging.error(f"An unexpected error occurred while accessing st.secrets: {e}")
        SOURCE_MESSAGE = f"Key Source: Streamlit Secrets (Error: {e})"

# --- Add Debugging Output Early --- 
st.sidebar.info(SOURCE_MESSAGE) # Show where the key was (or wasn't) found
if API_KEY_LOADED_SUCCESSFULLY:
    # Mask key for display
    masked_key = f"{PERPLEXITY_API_KEY[:7]}...{PERPLEXITY_API_KEY[-4:]}" if PERPLEXITY_API_KEY and len(PERPLEXITY_API_KEY) > 11 else "Invalid Key Format"
    st.sidebar.success(f"API Key Status: Loaded ({masked_key})")
else:
    st.sidebar.error("API Key Status: NOT loaded.")
# --- End Debugging Output ---

PERPLEXITY_API_BASE_URL = "https://api.perplexity.ai"

openai_client = None
client_init_error_msg = None

if PERPLEXITY_API_KEY:
    # Log the key just before use (masked)
    masked_key_for_log = f"{PERPLEXITY_API_KEY[:7]}...{PERPLEXITY_API_KEY[-4:]}" if PERPLEXITY_API_KEY and len(PERPLEXITY_API_KEY) > 11 else "Invalid Key Format"
    logging.info(f"Attempting to initialize OpenAI client with key: {masked_key_for_log}")
    try:
        # RE-ADD: Explicitly create an httpx client that ignores system proxies
        http_client = httpx.Client(proxies=None)
        
        # RE-ADD: Pass the custom http_client
        openai_client = OpenAI(
            api_key=PERPLEXITY_API_KEY, 
            base_url=PERPLEXITY_API_BASE_URL,
            http_client=http_client 
        )
        logging.info("OpenAI client initialized pointing to Perplexity API.")
        st.sidebar.success("API Client Status: Initialized.")
    except Exception as client_init_error:
        client_init_error_msg = str(client_init_error) 
        logging.error(f"Failed to initialize OpenAI client: {client_init_error_msg}", exc_info=True)
        openai_client = None 
        st.sidebar.error(f"API Client Status: Failed ({client_init_error_msg})")
else:
    st.sidebar.warning("API Client Status: Not initialized (No API Key).")

# Error message shown only if client is STILL None
if not openai_client:
    # Construct error message without accessing sidebar elements
    final_error_msg = "ERROR: Perplexity API client could not be initialized. "
    if not API_KEY_LOADED_SUCCESSFULLY:
        final_error_msg += f"API key was not loaded (checked {SOURCE_MESSAGE}). "
    elif client_init_error_msg:
        final_error_msg += f"Client initialization failed: {client_init_error_msg}. "
    else:
         final_error_msg += "Unknown initialization error. " # Fallback
    final_error_msg += "Please check API Key, app logs, and verify configuration."
    
    st.error(final_error_msg)
    st.stop()

NEGATIVE_KEYWORDS = '(arrest OR bankruptcy OR BSA OR conviction OR criminal OR fraud OR trafficking OR lawsuit OR "money laundering" OR OFAC OR Ponzi OR terrorist OR violation OR "honorary consul" OR consul OR "Panama Papers" OR theft OR corruption OR bribery)'
PERPLEXITY_MODEL = "sonar-pro"

# --- Core Functions (Adapted from Flask app) ---

def search_with_perplexity(company_name):
    # (This function remains largely the same as in app.py)
    # ... (API call logic, prompt, message structure) ...
    logging.info(f"Starting Perplexity search for company: {company_name}")
    if not openai_client:
        logging.error("OpenAI client (for Perplexity) not initialized.")
        return {"status": "failed", "error": "Perplexity API client not initialized.", "answer": None, "citations": [], "aml_grade": None}
    try:
        prompt = (
            f"First, on a single line, provide an Anti-Money Laundering (AML) risk grade for the company '{company_name}' based *only* on the negative news search results below. Use a scale from A (very low risk) to F (very high risk). Format this line ONLY as: 'AML Risk Grade: [GRADE]'. "
            f"\n\nThen, using **plain text only (no markdown)**, provide a brief summary of the company '{company_name}'. "
            f"\n\nAfter the summary, using **plain text only (no markdown)**, search for and summarize any negative news regarding this company, focusing *only* on the following keywords: {NEGATIVE_KEYWORDS}. "
            f"\n\nUse double line breaks to clearly separate the company summary from the negative news summary. Provide citations as numeric references like [1], [2] etc., within the text where applicable."
        )
        messages = [
            {"role": "system", "content": "You are an AI assistant performing company research... plain text only..."},
            {"role": "user", "content": prompt},
        ]
        logging.info(f"Calling Perplexity API (model: {PERPLEXITY_MODEL})...")
        response = openai_client.chat.completions.create(model=PERPLEXITY_MODEL, messages=messages, temperature=0.2)
        logging.info("Perplexity API call completed.")
        
        full_answer_content = None
        citations = []
        aml_grade = None
        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message
            if message and message.content:
                full_answer_content = message.content
                match = re.match(r"AML Risk Grade: ([A-F])", full_answer_content, re.IGNORECASE)
                if match:
                    aml_grade = match.group(1).upper()
                    full_answer_content = re.sub(r"AML Risk Grade: [A-F]\n*", "", full_answer_content, count=1, flags=re.IGNORECASE).strip()
                else:
                     logging.warning("Could not extract AML Grade.")
            # --- Citation Extraction (same as before) ---
            raw_citations = []
            # ... (check message.citations, response.citations) ...
            if hasattr(message, 'citations') and message.citations:
                 raw_citations = message.citations
            elif hasattr(response, 'citations') and response.citations:
                 raw_citations = response.citations
                 
            if raw_citations:
                 for cit in raw_citations:
                     # ... (standardize to dict) ...
                     citation_dict = {'url': '#', 'title': 'Source'}
                     if isinstance(cit, dict):
                         citation_dict['url'] = cit.get('url', '#')
                         citation_dict['title'] = cit.get('title', cit.get('url', 'Source'))
                     elif hasattr(cit, 'url'):
                          citation_dict['url'] = getattr(cit, 'url', '#')
                          citation_dict['title'] = getattr(cit, 'title', getattr(cit, 'url', 'Source'))
                     elif isinstance(cit, str):
                          citation_dict['url'] = cit
                          citation_dict['title'] = cit
                     else:
                          citation_dict['title'] = str(cit)
                     citations.append(citation_dict)
                     
        if not full_answer_content:
            full_answer_content = "No summary could be generated by Perplexity."
            
        return {"status": "success", "error": None, "answer": full_answer_content, "citations": citations, "aml_grade": aml_grade}

    except Exception as e:
        logging.error(f"Error during Perplexity search for {company_name}: {str(e)}", exc_info=True)
        return {"status": "failed", "error": str(e), "answer": None, "citations": [], "aml_grade": None}

def generate_pdf_bytes(company_name, data):
    """Generates the PDF content and returns it as bytes."""
    logging.info(f"Attempting to generate PDF bytes for {company_name}")
    buffer = io.BytesIO()
    try:
        doc = SimpleDocTemplate(buffer, pagesize=(8.5*inch, 11*inch), leftMargin=0.75*inch, rightMargin=0.75*inch, topMargin=0.75*inch, bottomMargin=0.75*inch)
        styles = getSampleStyleSheet()
        story = []

        # --- AML Grade (same styling logic) ---
        aml_grade = data.get("aml_grade", "N/A")
        grade_color = colors.grey
        # ... (grade color assignment) ...
        if aml_grade == 'A': grade_color = colors.darkgreen
        elif aml_grade == 'B': grade_color = colors.green
        elif aml_grade == 'C': grade_color = colors.orange
        elif aml_grade == 'D': grade_color = colors.orangered
        elif aml_grade == 'F': grade_color = colors.darkred
        grade_style = ParagraphStyle(name='AMLGrade', parent=styles['h1'], fontSize=28, textColor=grade_color, alignment=TA_CENTER, spaceAfter=15)
        story.append(Paragraph(f"AML Risk: {aml_grade}", grade_style))

        # --- Title (same styling logic) ---
        title_style = styles['h1']
        title_style.alignment = TA_CENTER
        title_style.fontSize = 18
        story.append(Paragraph(f"Research Report: {company_name}", title_style))
        story.append(Spacer(1, 0.2*inch))

        # --- Summary & Analysis (Plain Text Rendering - same logic) ---
        body_style = ParagraphStyle(name='BodyText', parent=styles['Normal'], spaceBefore=6, spaceAfter=6, leading=14, fontSize=10, alignment=TA_LEFT)
        answer_text = data.get("answer", "N/A")
        escaped_answer = answer_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        story.append(Paragraph(f'<pre>{escaped_answer}</pre>', body_style))
        story.append(Spacer(1, 0.2*inch))

        # --- Citations Section (same styling logic) ---
        story.append(Paragraph("Sources Cited", styles['h2']))
        story.append(Spacer(1, 0.1*inch))
        citations = data.get("citations", [])
        if citations:
            citation_style = ParagraphStyle(name='Citation', parent=styles['Normal'], fontSize=9, leading=11, spaceAfter=4)
            for i, citation in enumerate(citations):
                url = citation.get('url', '#')
                title = citation.get('title', url)
                safe_title = title.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                safe_url = url.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                citation_text = f'{i+1}. <a href="{safe_url}" color="blue"><u>{safe_title}</u></a>'
                story.append(Paragraph(citation_text, citation_style))
        else:
            story.append(Paragraph("None provided or embedded in text.", styles['Italic']))

        # Build the PDF in the buffer
        doc.build(story)
        logging.info(f"Successfully generated PDF bytes for {company_name}")
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as pdf_error:
        logging.error(f"Error generating PDF bytes for {company_name}: {pdf_error}", exc_info=True)
        return None

# --- Streamlit App UI ---
st.title("Bulk Company Research Tool")
st.markdown("Enter company names (one per line) to generate PDF research reports including summaries, negative news analysis (based on specific keywords), and an AML risk grade.")

company_names_input = st.text_area("Company Names (one per line)", height=150, placeholder="e.g.\nGoogle\nMicrosoft\nNonExistent Company Example")

start_button = st.button("Generate PDF Reports")

if start_button and company_names_input:
    company_names = [name.strip() for name in company_names_input.split('\n') if name.strip()]
    st.info(f"Processing {len(company_names)} company name(s)... Please wait.")
    
    # Use columns for better layout of results
    col1, col2 = st.columns([3, 1]) # Company name/status in first col, download button in second
    
    with col1:
        st.subheader("Processing Status:")
        
    with col2:
        st.write("") # Placeholder for alignment

    results_placeholder = st.empty()
    results_list = []

    progress_bar = st.progress(0)
    total_names = len(company_names)

    for i, name in enumerate(company_names):
        with st.spinner(f"Researching {name}..."):
            search_data = search_with_perplexity(name)
            pdf_bytes = None
            status = search_data["status"]
            error_message = search_data["error"]
            
            if status == "success":
                pdf_bytes = generate_pdf_bytes(name, search_data)
                if pdf_bytes is None:
                    status = "failed"
                    error_message = "PDF generation failed."
            
            results_list.append({
                'name': name,
                'status': status,
                'error_message': error_message,
                'pdf_bytes': pdf_bytes
            })
            
        # Update progress bar
        progress_bar.progress((i + 1) / total_names)

    # Display results with download buttons after processing all
    st.success("Processing Complete!")
    results_placeholder.empty() # Clear the overall status message if needed
    
    st.divider()
    st.subheader("Download Reports:")
    
    cols = st.columns(2) # Create two columns for results
    current_col = 0
    
    for result in results_list:
        with cols[current_col]:
            if result['status'] == 'success' and result['pdf_bytes']:
                safe_name = "".join(c if c.isalnum() else '_' for c in result['name'])
                st.markdown(f"**{result['name']}**")
                st.download_button(
                    label=f"Download PDF",
                    data=result['pdf_bytes'],
                    file_name=f"{safe_name}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    key=f"download_{safe_name}_{i}" # Unique key for each button
                )
            else:
                st.error(f"**{result['name']}**: Failed ({result.get('error_message', 'Unknown error')})")
            st.markdown("&nbsp;") # Add a little space below each item
            
        current_col = 1 - current_col # Alternate columns
        
elif start_button and not company_names_input:
    st.warning("Please enter at least one company name.") 