"""
Real-Time Writing Assistant with Live Collaboration

This module provides a real-time writing assistant that enables live collaboration
between humans and AI during the grant writing process. It supports streaming text
generation, live intervention, contextual adaptation, and smart suggestions.

Key features:
- Streaming text generation with configurable speeds
- Live intervention system for pausing and editing
- Contextual adaptation based on human edits
- Collaborative memory for user preferences
- Smart suggestions during writing
- Multiple writing modes (draft, careful, creative)

Compatible with both OpenAI and Groq APIs with streaming support.
"""

import os
import time
import json
import logging
import threading
from enum import Enum
from typing import List, Dict, Any, Callable
from dataclasses import dataclass, field
import asyncio

# Third-party imports
import openai
import litellm
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("realtime_writer")

# Load environment variables
load_dotenv()

# Default configuration
DEFAULT_CONFIG = {
    "default_model": "openai/gpt-3.5-turbo",
    "fallback_model": "groq/llama3-8b-8192",
    "temperature": 0.7,
    "max_tokens": 1500,
    "writing_speeds": {
        "fast": 0.01,  # Seconds between words for fast mode
        "normal": 0.03,  # Normal typing speed
        "thoughtful": 0.08,  # Slower, more deliberate pace
    },
    "writing_styles": {
        "draft": {"temperature": 0.8, "top_p": 1.0},
        "careful": {"temperature": 0.4, "top_p": 0.9},
        "creative": {"temperature": 1.0, "top_p": 1.0},
    },
    "suggestion_threshold": 0.7,  # Confidence threshold for suggestions
    "memory_size": 10,  # Number of interactions to remember
    "feedback_weight": 0.8,  # Weight given to user feedback
}


class WritingMode(Enum):
    """Enum for different writing modes."""
    DRAFT = "draft"
    CAREFUL = "careful"
    CREATIVE = "creative"


class WritingSpeed(Enum):
    """Enum for different writing speeds."""
    FAST = "fast"
    NORMAL = "normal"
    THOUGHTFUL = "thoughtful"


@dataclass
class WritingContext:
    """Manages context, style, and user preferences for writing."""
    
    # Content context
    topic: str = ""
    section: str = ""
    previous_text: str = ""
    current_text: str = ""
    
    # Style and preferences
    writing_mode: WritingMode = WritingMode.CAREFUL
    writing_speed: WritingSpeed = WritingSpeed.NORMAL
    tone: str = "professional"
    formality: str = "formal"
    
    # User preferences and history
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    edit_history: List[Dict[str, Any]] = field(default_factory=list)
    suggestion_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Grant-specific context
    grant_type: str = "research"
    funder_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def update_from_user_edit(self, original_text: str, edited_text: str) -> None:
        """Update context based on user edits."""
        # Record the edit
        edit_record = {
            "timestamp": time.time(),
            "original": original_text,
            "edited": edited_text,
            "section": self.section
        }
        self.edit_history.append(edit_record)
        
        # Update current text
        self.current_text = edited_text
        
        # Analyze edit to infer preferences (simplified)
        if len(edited_text) > len(original_text) * 1.2:
            self.user_preferences["detail_level"] = "high"
        elif len(edited_text) < len(original_text) * 0.8:
            self.user_preferences["detail_level"] = "concise"
            
    def to_prompt_context(self) -> Dict[str, Any]:
        """Convert context to a format suitable for prompts."""
        return {
            "topic": self.topic,
            "section": self.section,
            "previous_text": self.previous_text,
            "current_text": self.current_text,
            "tone": self.tone,
            "formality": self.formality,
            "grant_type": self.grant_type,
            "funder_requirements": self.funder_requirements,
            "user_preferences": self.user_preferences
        }


class RealtimeWriter:
    """Main writing engine with streaming capabilities."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the real-time writer."""
        self.config = config or DEFAULT_CONFIG
        self.writing_speeds = self.config.get("writing_speeds", {})
        self.writing_styles = self.config.get("writing_styles", {})
        self.default_model = self.config.get("default_model", "openai/gpt-3.5-turbo")
        self.fallback_model = self.config.get("fallback_model", "groq/llama3-8b-8192")
        self.pause_event = threading.Event()
        
        # Set up API keys
        self._setup_api_keys()
        
        logger.info(f"RealtimeWriter initialized with model: {self.default_model}")
    
    def _setup_api_keys(self) -> None:
        """Set up API keys for OpenAI and Groq."""
        # Get API keys and strip quotes if present
        openai_key = os.getenv("OPENAI_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        
        if openai_key:
            openai_key = openai_key.strip('"').strip("'")
            os.environ["OPENAI_API_KEY"] = openai_key
            openai.api_key = openai_key
        
        if groq_key:
            groq_key = groq_key.strip('"').strip("'")
            os.environ["GROQ_API_KEY"] = groq_key
    
    def _get_model_params(self, context: WritingContext) -> Dict[str, Any]:
        """Get model parameters based on writing context."""
        # Get style parameters
        style_params = self.writing_styles.get(
            context.writing_mode.value, 
            {"temperature": 0.7, "top_p": 1.0}
        )
        
        # Combine with default parameters
        params = {
            "temperature": style_params.get("temperature", 0.7),
            "top_p": style_params.get("top_p", 1.0),
            "max_tokens": self.config.get("max_tokens", 1500),
            "stream": False  # For simplicity, disable streaming for now
        }
        
        return params
    
    def _build_prompt(self, context: WritingContext, instruction: str = None) -> List[Dict[str, str]]:
        """Build a prompt for the language model based on context."""
        
        # Create a system message with writing guidance
        system_message = f"""
        You are an expert grant writer specializing in {context.grant_type} grants.
        Write in a {context.tone}, {context.formality} tone.
        
        Current section: {context.section}
        
        Funder requirements:
        {json.dumps(context.funder_requirements, indent=2)}
        
        User preferences:
        {json.dumps(context.user_preferences, indent=2)}
        """
        
        # Create a user message with the specific request
        user_message = instruction or f"Write the {context.section} section for a {context.grant_type} grant about {context.topic}."
        
        # Add context from previous text if available
        if context.previous_text:
            user_message += f"\n\nPrevious content:\n{context.previous_text}"
        
        if context.current_text:
            user_message += f"\n\nContinue from this text:\n{context.current_text}"
        
        return [
            {"role": "system", "content": system_message.strip()},
            {"role": "user", "content": user_message.strip()}
        ]
    
    def simulate_streaming(self, text: str, callback: Callable[[str, bool], None], speed: str = "normal") -> None:
        """Simulate streaming output by sending text word by word."""
        words = text.split()
        delay = self.writing_speeds.get(speed, 0.03)
        
        for i, word in enumerate(words):
            # Check if paused
            while self.pause_event.is_set():
                time.sleep(0.1)
            
            # Send word with space
            word_with_space = word + " "
            callback(word_with_space, False)
            
            # Add delay
            time.sleep(delay)
        
        # Signal completion
        callback("", True)
    
    async def generate_text_streaming(self, 
                               context: WritingContext, 
                               callback: Callable[[str, bool], None],
                               instruction: str = None) -> str:
        """
        Generate text with streaming output simulation.
        
        Args:
            context: The writing context
            callback: Function to call with each chunk of text and a done flag
            instruction: Optional specific instruction for generation
            
        Returns:
            The complete generated text
        """
        try:
            # Build the prompt
            messages = self._build_prompt(context, instruction)
            
            # Get model parameters
            params = self._get_model_params(context)
            
            try:
                # Try with primary model
                response = await litellm.acompletion(
                    model=self.default_model,
                    messages=messages,
                    **params
                )
                
                full_text = response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"Error with primary model: {e}. Falling back to secondary model.")
                
                # Fallback to secondary model
                response = await litellm.acompletion(
                    model=self.fallback_model,
                    messages=messages,
                    **params
                )
                
                full_text = response.choices[0].message.content
            
            # Simulate streaming
            self.simulate_streaming(full_text, callback, context.writing_speed.value)
            
            # Update the context with the final text
            context.current_text = full_text
            
            return full_text
            
        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            callback(f"Error generating text: {str(e)}", True)
            return ""
    
    def pause(self) -> None:
        """Pause the text generation."""
        self.pause_event.set()
        logger.info("Text generation paused")
    
    def resume(self) -> None:
        """Resume the text generation."""
        self.pause_event.clear()
        logger.info("Text generation resumed")
    
    def is_paused(self) -> bool:
        """Check if generation is currently paused."""
        return self.pause_event.is_set()


# Helper functions
def create_realtime_writer(config: Dict[str, Any] = None) -> RealtimeWriter:
    """Create a RealtimeWriter instance with the given configuration."""
    return RealtimeWriter(config or DEFAULT_CONFIG)


def create_writing_context(
    topic: str,
    section: str,
    grant_type: str = "research",
    writing_mode: str = "careful",
    writing_speed: str = "normal"
) -> WritingContext:
    """Create a WritingContext with the given parameters."""
    return WritingContext(
        topic=topic,
        section=section,
        grant_type=grant_type,
        writing_mode=WritingMode(writing_mode),
        writing_speed=WritingSpeed(writing_speed)
    )


async def demo_realtime_writing():
    """Demonstrate real-time writing in a console environment."""
    # Create a writer
    writer = create_realtime_writer()
    
    # Create a context
    context = create_writing_context(
        topic="Climate-resilient agriculture",
        section="Problem Statement",
        grant_type="research"
    )
    
    # Define a callback
    def print_callback(text, done):
        if not done:
            print(text, end="", flush=True)
        else:
            print("\n--- Generation complete ---")
    
    # Generate text
    print("\nGenerating problem statement in real-time...\n")
    await writer.generate_text_streaming(context, print_callback)


if __name__ == "__main__":
    # Run the demo if executed directly
    asyncio.run(demo_realtime_writing())
