"""
Real-Time Grant Writing Assistant

A Streamlit application that enables live collaboration between humans and AI
during the grant writing process. This application demonstrates the human-in-the-loop
approach where users can intervene, guide, and refine the AI's writing in real-time.
"""

import os
import time
import asyncio
import threading
from typing import Dict, Any, List
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
import plotly.graph_objects as go

# Import the RealtimeWriter and related classes
from utils.realtime_writer import (
    RealtimeWriter, 
    WritingContext,
    WritingMode,
    WritingSpeed,
    DEFAULT_CONFIG
)

# Configure page settings
st.set_page_config(
    page_title="Real-Time Grant Writing Assistant",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    st.markdown("""
    <style>
    .main {
        background-color: #F5F7F9;
    }
    .stTextArea textarea {
        font-family: 'Roboto', sans-serif;
        font-size: 16px;
        line-height: 1.6;
    }
    .status-indicator {
        font-weight: 500;
        padding: 5px 10px;
        border-radius: 15px;
        display: inline-block;
    }
    .status-writing {
        background-color: #E3F2FD;
        color: #1565C0;
    }
    .status-paused {
        background-color: #FFF8E1;
        color: #FF8F00;
    }
    .status-complete {
        background-color: #E0F2F1;
        color: #00796B;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables if they don't exist."""
    if "writer" not in st.session_state:
        st.session_state.writer = RealtimeWriter()
    
    if "writing_context" not in st.session_state:
        st.session_state.writing_context = WritingContext(
            topic="",
            section="",
            grant_type="research",
            writing_mode=WritingMode.CAREFUL,
            writing_speed=WritingSpeed.NORMAL,
        )
    
    if "current_text" not in st.session_state:
        st.session_state.current_text = ""
    
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False
    
    if "word_count" not in st.session_state:
        st.session_state.word_count = 0
    
    if "generation_complete" not in st.session_state:
        st.session_state.generation_complete = False

def update_word_count(text: str) -> int:
    """Update the word count based on the current text."""
    words = len(text.split())
    st.session_state.word_count = words
    return words

def text_callback(text: str, done: bool) -> None:
    """Callback function for the streaming text generation."""
    if not done:
        # Update the current text
        st.session_state.current_text += text
        
        # Update the word count
        update_word_count(st.session_state.current_text)
        
        # Rerun to update the UI
        st.rerun()
    else:
        # Generation is complete
        st.session_state.is_generating = False
        st.session_state.generation_complete = True
        
        # Update the writing context
        st.session_state.writing_context.current_text = st.session_state.current_text
        
        # Rerun to update the UI
        st.rerun()

async def start_generation():
    """Start the text generation process."""
    try:
        # Reset state
        st.session_state.is_generating = True
        st.session_state.generation_complete = False
        
        # Get the context
        context = st.session_state.writing_context
        
        # Start generation
        await st.session_state.writer.generate_text_streaming(
            context=context,
            callback=text_callback
        )
        
    except Exception as e:
        st.error(f"Error during text generation: {e}")
        st.session_state.is_generating = False

def start_generation_thread():
    """Start text generation in a separate thread."""
    # Create and start the thread
    thread = threading.Thread(target=lambda: asyncio.run(start_generation()))
    add_script_run_ctx(thread)  # Add Streamlit context to the thread
    thread.start()

def pause_generation():
    """Pause the text generation."""
    st.session_state.writer.pause()

def resume_generation():
    """Resume the text generation."""
    st.session_state.writer.resume()

def create_progress_chart():
    """Create a simple progress chart."""
    # Simple progress based on word count
    target_words = 300  # Target word count for the section
    progress = min(100, int((st.session_state.word_count / target_words) * 100))
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = progress,
        title = {'text': "Section Progress (%)"},
        delta = {'reference': 100},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(height=300)
    return fig

# Main application
def main():
    # Load CSS
    load_css()
    
    # Initialize session state
    init_session_state()
    
    # Create sidebar
    with st.sidebar:
        st.title("📝 Real-Time Writer")
        
        # Project settings
        st.header("Project Settings")
        
        # Grant type
        grant_type = st.selectbox(
            "Grant Type",
            ["Research", "Non-Profit", "Educational", "Technology", "Healthcare"],
            index=0
        )
        
        # Topic
        topic = st.text_input(
            "Topic",
            value=st.session_state.writing_context.topic,
            help="The main subject of your grant proposal"
        )
        
        # Section
        section = st.selectbox(
            "Current Section",
            ["Problem Statement", "Methodology", "Expected Outcomes", "Budget"],
            index=0
        )
        
        # Writing mode
        st.header("Writing Style")
        
        col1, col2 = st.columns(2)
        
        with col1:
            writing_mode = st.selectbox(
                "Mode",
                ["draft", "careful", "creative"],
                index=1
            )
        
        with col2:
            writing_speed = st.selectbox(
                "Speed",
                ["fast", "normal", "thoughtful"],
                index=1
            )
        
        # Update the writing context
        if (topic != st.session_state.writing_context.topic or
            section != st.session_state.writing_context.section or
            grant_type.lower() != st.session_state.writing_context.grant_type or
            writing_mode != st.session_state.writing_context.writing_mode.value or
            writing_speed != st.session_state.writing_context.writing_speed.value):
            
            # Update the context
            st.session_state.writing_context.topic = topic
            st.session_state.writing_context.section = section
            st.session_state.writing_context.grant_type = grant_type.lower()
            st.session_state.writing_context.writing_mode = WritingMode(writing_mode)
            st.session_state.writing_context.writing_speed = WritingSpeed(writing_speed)
    
    # Main content area
    st.title("🚀 Real-Time Grant Writing Assistant")
    st.markdown("*Human-in-the-Loop Collaborative Writing*")
    
    # Status indicator
    status_text = "Ready"
    status_class = ""
    
    if st.session_state.is_generating and not st.session_state.writer.is_paused():
        status_text = "AI Writing..."
        status_class = "status-writing"
    elif st.session_state.writer.is_paused():
        status_text = "Paused"
        status_class = "status-paused"
    elif st.session_state.generation_complete:
        status_text = "Complete"
        status_class = "status-complete"
    
    # Display status
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <div>
            <span class="status-indicator {status_class}">{status_text}</span>
            <span style="margin-left: 15px; color: #757575;">
                {st.session_state.word_count} words | 
                Section: {st.session_state.writing_context.section} | 
                Type: {st.session_state.writing_context.grant_type.capitalize()}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Control buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not st.session_state.is_generating:
            if st.button("🚀 Start Writing", key="start_button", use_container_width=True):
                start_generation_thread()
        else:
            if st.session_state.writer.is_paused():
                if st.button("▶️ Resume", key="resume_button", use_container_width=True):
                    resume_generation()
            else:
                if st.button("⏸️ Pause", key="pause_button", use_container_width=True):
                    pause_generation()
    
    with col2:
        if st.button("🧹 Clear Text", key="clear_button", use_container_width=True):
            st.session_state.current_text = ""
            st.session_state.writing_context.current_text = ""
            update_word_count("")
    
    with col3:
        if st.button("📊 Show Progress", key="progress_button", use_container_width=True):
            st.balloons()
    
    with col4:
        if st.button("💾 Save Session", key="save_button", use_container_width=True):
            st.success("Session saved! (placeholder)")
    
    # Main content and progress in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### {st.session_state.writing_context.section}")
        
        # Text area for writing/editing
        current_text = st.text_area(
            "Grant Text",
            value=st.session_state.current_text,
            height=400,
            key="text_area",
            label_visibility="collapsed"
        )
        
        # Handle edits
        if current_text != st.session_state.current_text:
            st.session_state.current_text = current_text
            update_word_count(current_text)
            st.session_state.writing_context.current_text = current_text
        
        # Word count and progress
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; margin-top: 10px;">
            <span>{st.session_state.word_count} words</span>
            <span>Target: 300 words</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar
        target_words = 300
        progress = min(1.0, st.session_state.word_count / target_words)
        st.progress(progress)
    
    with col2:
        # Progress chart
        st.markdown("### Progress Tracking")
        progress_chart = create_progress_chart()
        st.plotly_chart(progress_chart, use_container_width=True)
        
        # Real-time features info
        st.markdown("### 🔄 Real-Time Features")
        st.info("""
        ✅ **Live Writing**: AI types word-by-word  
        ⏸️ **Pause/Resume**: Intervene anytime  
        ✏️ **Live Editing**: Edit while AI writes  
        📊 **Progress Tracking**: Visual feedback  
        💡 **Context Aware**: AI adapts to your style
        """)
        
        # Writing tips
        st.markdown("### 💡 Tips")
        st.markdown("""
        - **Pause** to provide direction
        - **Edit** text to guide AI style
        - **Different modes**: draft, careful, creative
        - **Speed control**: fast, normal, thoughtful
        - **Save sessions** for later
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #757575; font-size: 14px;">
        Real-Time Grant Writing Assistant | Human-in-the-Loop Collaboration | v1.0
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
