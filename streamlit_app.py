import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
from datetime import datetime
import logging
import re
import io # For handling PDF data in memory

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

# Try environment variable first (ideal for local with .env)
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
SECRETS_TRIED = False

if PERPLEXITY_API_KEY:
    logging.info("Loaded API key from environment variable.")
else:
    # If env var not found, try st.secrets (for deployment)
    logging.info("API key not in environment variables, trying st.secrets...")
    SECRETS_TRIED = True
    try:
        PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]
        logging.info("Loaded API key from st.secrets")
    except (FileNotFoundError, KeyError):
        # If secrets also fail, log the final warning
        logging.warning("API key not found in st.secrets or environment variables.")
        PERPLEXITY_API_KEY = None # Ensure it's None if both failed
    except Exception as e:
        # Catch any other potential st.secrets errors
        logging.error(f"An unexpected error occurred while accessing st.secrets: {e}")
        PERPLEXITY_API_KEY = None

PERPLEXITY_API_BASE_URL = "https://api.perplexity.ai"

openai_client = None
if PERPLEXITY_API_KEY:
    try:
        openai_client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url=PERPLEXITY_API_BASE_URL)
        logging.info("OpenAI client initialized pointing to Perplexity API.")
    except Exception as client_init_error:
        logging.error(f"Failed to initialize OpenAI client: {client_init_error}", exc_info=True)
        openai_client = None # Ensure client is None if init fails

# Error message shown only if API key is STILL None after trying both methods
if not openai_client: 
    # Construct message based on whether secrets were attempted
    error_msg_detail = "(tried environment variables and Streamlit secrets)" if SECRETS_TRIED else "(tried environment variable)"
    st.error(f"Perplexity API key not found {error_msg_detail}. Please set it in your .env file locally or Streamlit secrets for deployment.")

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

if not openai_client:
    st.stop() # Stop execution if API client failed to initialize

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