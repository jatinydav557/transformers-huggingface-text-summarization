import streamlit as st
from src.textSummarizer.pipeline.prediction_pipeline import PredictionPipeline

# Page configuration
st.set_page_config(
    page_title="Text Summarizer ✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling and layout
st.markdown("""
    <style>
        /* Import Google Font - Poppins */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        /* Apply Poppins font and text color globally */
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
            color: #f1f1f1; /* Light text color for contrast */
        }

        /* Background image for the entire application, applied to html and body */
        html, body {
            background-image: url("https://images.unsplash.com/photo-1549429440-621b1ed1d471?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"); /* Direct image URL for Northern Lights */
            background-size: cover; /* Cover the entire area */
            background-position: center; /* Center the image */
            background-repeat: no-repeat; /* Do not repeat the image */
            background-attachment: fixed; /* Keep image fixed on scroll */
            margin: 0; /* Remove default body margin */
            padding: 0; /* Remove default body padding */
            overflow-x: hidden; /* Prevent horizontal scrollbar */
        }

        /* Ensure .stApp fills the viewport and its content is centered */
        .stApp {
            display: flex; /* Use flexbox for main app layout */
            flex-direction: column; /* Stack content vertically */
            align-items: center; /* Center content horizontally */
            min-height: 100vh; /* Ensure app takes full viewport height */
            padding-top: 50px; /* Padding from the top of the screen */
            padding-bottom: 50px; /* Padding from the bottom */
            background-color: transparent; /* Ensure no conflicting background */
        }

        /* Adjust Streamlit's default main content container to align properly */
        .stApp > div:first-child {
            width: 100%; /* Take full width of parent (flex container) */
            max-width: 800px; /* Max width for better readability on large screens */
            padding: 0 15px; /* Horizontal padding for smaller screens */
            box-sizing: border-box; /* Include padding in element's total width */
        }

        /* Styling for the main content card (input, button, summary) */
        .content-card {
            background: rgba(255, 255, 255, 0.1); /* Semi-transparent background */
            padding: 2rem; /* Inner padding */
            border-radius: 15px; /* Rounded corners */
            box-shadow: 0 8px 32px 0 rgba( 31, 38, 135, 0.37 ); /* Glassmorphism shadow */
            backdrop-filter: blur(10px); /* Blur effect */
            -webkit-backdrop-filter: blur(10px); /* For Safari compatibility */
            border: 1px solid rgba(255, 255, 255, 0.18); /* Subtle border */
            margin-top: 20px; /* Space after titles */
            width: 100%; /* Take full width of its parent */
            max-width: 700px; /* Maximum width for the card itself */
            display: flex; /* Use flexbox for internal elements */
            flex-direction: column; /* Stack vertically */
            gap: 1rem; /* Space between elements within the card */
        }

        /* Style for the text input area */
        .stTextArea textarea {
            border-radius: 10px; /* Rounded corners */
            padding: 1rem; /* Inner padding */
            font-size: 1rem; /* Font size */
            background-color: rgba(255,255,255,0.9); /* Opaque white background */
            color: #000; /* Black text color */
        }

        /* Style for buttons */
        .stButton button {
            background-color: #ff4b2b; /* Vibrant red-orange background */
            color: white; /* White text */
            padding: 0.75rem 1.5rem; /* Padding */
            font-weight: bold; /* Bold text */
            border: none; /* No border */
            border-radius: 8px; /* Rounded corners */
            transition: background 0.3s ease-in-out; /* Smooth hover transition */
            cursor: pointer; /* Pointer cursor on hover */
            align-self: flex-start; /* Align button to the start (left) */
        }
        .stButton button:hover {
            background-color: #ff6a3d; /* Lighter red-orange on hover */
        }

        /* Style for the summary output box */
        .summary-box {
            margin-top: 20px; /* Space above the summary box */
            background-color: #0f2027; /* Darker background for summary */
            padding: 1.5rem; /* Inner padding */
            border-left: 5px solid #ff4b2b; /* Left border accent */
            border-radius: 10px; /* Rounded corners */
            white-space: pre-wrap; /* Preserve whitespace and wrap text */
        }

        /* Hide Streamlit's default header, footer, and menu button */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Custom styling for Streamlit's generated H1 and H2 elements */
        h1.st-emotion-cache-xyz, h2.st-emotion-cache-xyz {
            text-align: center; /* Center the text */
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5); /* Text shadow for readability */
        }
        h1.st-emotion-cache-xyz {
            font-size: 3em; /* Larger font for main title */
            margin-bottom: 0.2em; /* Space below title */
        }
        h2.st-emotion-cache-xyz {
            font-size: 1.5em; /* Smaller font for subtitle */
            margin-top: 0; /* No top margin */
            margin-bottom: 1em; /* Space below subtitle */
            opacity: 0.9; /* Slightly transparent */
        }

        /* Ensure the main content block also centers its children */
        .block-container {
            display: flex;
            flex-direction: column;
            align-items: center; /* Center content horizontally */
            width: 100%;
        }

    </style>
""", unsafe_allow_html=True)

# Main title and subtitle are placed directly on the background
st.title("🧠 Text Summarizer")
st.subheader("Summarize long paragraphs into short, meaningful summaries ✨")

# The content card wraps the input, button, and summary output
st.markdown("<div class='content-card'>", unsafe_allow_html=True)

# Text area for user input
text_input = st.text_area("Enter your text below 👇", height=250)

# Summarize button
if st.button("🔍 Summarize"):
    if text_input.strip(): # Check if input is not empty
        with st.spinner("Generating summary..."): # Show spinner during processing
            obj = PredictionPipeline() # Initialize prediction pipeline
            summary = obj.predict(text_input) # Get summary from the model

        # Display the summary in a styled box
        st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
        st.subheader("Summary 📝")
        st.write(summary)
        st.markdown("</div>", unsafe_allow_html=True)

        # Download summary button
        st.download_button(
            label="📥 Download Summary",
            data=summary,
            file_name="summary.txt",
            mime="text/plain",
            key="download_summary_button" # Unique key for the button
        )
    else:
        st.warning("⚠️ Please enter some text first.") # Warning if input is empty

st.markdown("</div>", unsafe_allow_html=True) # Close the content-card div
