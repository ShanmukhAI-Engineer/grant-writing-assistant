"""
Enhanced Sample Integration

This module demonstrates how to integrate the enhanced sample processor
with the existing CrewAI workflow to create a hybrid approach that combines
few-shot examples from various document types (PDF, Word, text) with RAG.

Usage:
    python enhanced_sample_integration.py --sample_dir data/samples --grant_type research

Integration points:
1. Sample processing from multiple file types
2. CrewAI prompt enhancement with sample-based style matching
3. RAG system integration for additional context
4. Hybrid approach combining few-shot examples with RAG retrieval
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import time
import json
import tempfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules
try:
    from utils.sample_processor import SampleGrantProcessor
    from utils.rag_utils import load_and_store_pdf, query_context, VECTOR_DB_DIR
    from crew_agents.crew_builder import setup_api_keys
except ImportError as e:
    logger.error(f"Failed to import project modules: {e}")
    logger.error(
        "Make sure you're running this script from the project root directory")
    sys.exit(1)


class HybridGrantGenerator:
    """
    Integrates the enhanced sample processor with CrewAI and RAG
    to generate grants using a hybrid approach.
    """

    def __init__(
        self,
        sample_dir: str = "data/samples",
        vector_db_dir: str = VECTOR_DB_DIR,
        model_provider: str = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the hybrid grant generator.

        Args:
            sample_dir: Directory containing grant samples
            vector_db_dir: Directory for the vector database
            model_provider: LLM provider (openai or groq)
            config: Additional configuration options
        """
        self.sample_dir = Path(sample_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.config = config or {}

        # Set default configuration values
        self.config.setdefault("batch_size", 5)
        self.config.setdefault("use_rag", True)
        self.config.setdefault("use_samples", True)
        self.config.setdefault("max_rag_results", 3)
        self.config.setdefault("max_sample_length", 800)
        self.config.setdefault("similarity_threshold", 0.7)

        # Initialize components
        self._initialize_components(model_provider)

        logger.info(
            f"Initialized HybridGrantGenerator with config: {json.dumps(self.config, indent=2)}")

    def _initialize_components(self, model_provider: str = None) -> None:
        """Initialize the sample processor and other components."""
        # Setup API keys and model
        try:
            self.model = model_provider or setup_api_keys()
            logger.info(f"Using model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to setup API keys: {e}")
            logger.warning(
                "Continuing without LLM access - some features may be limited")
            self.model = None

        # Initialize sample processor
        try:
            self.sample_processor = SampleGrantProcessor(
                samples_dir=self.sample_dir)
            self.samples = self.sample_processor.load_samples(
                show_progress=True)

            stats = self.sample_processor.get_sample_statistics()
            logger.info(
                f"Loaded {stats['total_samples']} samples from {len(stats['by_type'])} grant types")
            logger.info(
                f"File types: {', '.join(stats['by_file_type'].keys())}")
        except Exception as e:
            logger.error(f"Failed to initialize sample processor: {e}")
            self.sample_processor = None
            self.samples = {}

            # If samples are required but unavailable, raise the error
            if self.config["use_samples"]:
                raise RuntimeError(
                    f"Sample processing is required but failed: {e}")

    def add_samples_to_rag(self) -> bool:
        """Add all loaded samples to the RAG system."""
        if not self.sample_processor or not self.samples:
            logger.warning("No samples available to add to RAG")
            return False

        if not self.config["use_rag"]:
            logger.info("RAG integration disabled in config")
            return False

        try:
            logger.info("Adding samples to RAG system...")
            # The method handles importing rag_utils
            self.sample_processor.add_sample_to_rag(None)
            logger.info("Successfully added samples to RAG")
            return True
        except Exception as e:
            logger.error(f"Failed to add samples to RAG: {e}")
            return False

    def create_hybrid_prompt(
        self,
        grant_type: str,
        topic: str,
        section: str,
        additional_context: str = None
    ) -> Dict[str, Any]:
        """
        Create a hybrid prompt combining sample-based style with RAG retrieval.

        Args:
            grant_type: Type of grant (research, nonprofit, etc.)
            topic: The grant topic
            section: The section to generate (e.g., "Problem Statement")
            additional_context: Any additional context to include

        Returns:
            Dict containing the prompt and metadata
        """
        start_time = time.time()
        result = {
            "prompt": "",
            "sample_used": None,
            "rag_results": [],
            "error": None,
            "generation_time": 0
        }

        # Step 1: Get sample-based prompt if enabled
        sample_prompt = ""
        if self.config["use_samples"] and self.sample_processor:
            try:
                sample = self.sample_processor.get_best_sample(
                    grant_type, topic)
                if sample:
                    result["sample_used"] = {
                        "filename": sample["filename"],
                        "file_type": sample.get("file_type", "unknown"),
                        "word_count": sample["word_count"]
                    }
                    sample_prompt = self.sample_processor.create_style_prompt(
                        sample, topic, section)
                    logger.info(
                        f"Created sample-based prompt using {sample['filename']}")
                else:
                    logger.warning(f"No samples found for {grant_type}")
            except Exception as e:
                logger.error(f"Error creating sample-based prompt: {e}")
                result["error"] = f"Sample processing error: {str(e)}"

        # Step 2: Get RAG context if enabled
        rag_context = ""
        if self.config["use_rag"]:
            try:
                # Create a query based on the topic and section
                query = f"{section} for {grant_type} grant about {topic}"

                # Get relevant context from RAG
                rag_results = query_context(
                    query,
                    k=self.config["max_rag_results"]
                )

                if rag_results:
                    # Extract and format the context
                    for i, doc in enumerate(rag_results):
                        result["rag_results"].append({
                            "content_preview": doc.page_content[:100] + "...",
                            "metadata": doc.metadata
                        })
                        rag_context += f"\nRELEVANT CONTEXT {i+1}:\n{doc.page_content}\n"

                    logger.info(
                        f"Retrieved {len(rag_results)} relevant documents from RAG")
                else:
                    logger.warning("No relevant context found in RAG")
            except Exception as e:
                logger.error(f"Error retrieving RAG context: {e}")
                result["error"] = f"RAG retrieval error: {str(e)}"

        # Step 3: Combine sample prompt with RAG context
        if sample_prompt and rag_context:
            # Hybrid approach: combine sample-based style with RAG context
            result["prompt"] = f"""
            {sample_prompt}
            
            Additionally, consider the following relevant information:
            {rag_context}
            
            {additional_context or ""}
            """
            logger.info(
                "Created hybrid prompt combining sample style with RAG context")
        elif sample_prompt:
            # Sample-only approach
            result["prompt"] = f"""
            {sample_prompt}
            
            {additional_context or ""}
            """
            logger.info("Created sample-based prompt (RAG unavailable)")
        elif rag_context:
            # RAG-only approach
            result["prompt"] = f"""
            Write a {section} for a {grant_type} grant proposal about "{topic}".
            
            Use the following relevant information:
            {rag_context}
            
            {additional_context or ""}
            """
            logger.info("Created RAG-based prompt (samples unavailable)")
        else:
            # Fallback to basic prompt
            result["prompt"] = f"""
            Write a {section} for a {grant_type} grant proposal about "{topic}".
            
            {additional_context or ""}
            """
            logger.warning(
                "Using fallback prompt (no samples or RAG available)")

        # Record generation time
        result["generation_time"] = time.time() - start_time

        return result

    def enhance_crew_tasks(
        self,
        crew_builder,
        grant_type: str,
        topic: str
    ) -> Dict[str, Any]:
        """
        Enhance CrewAI tasks with hybrid prompts.

        Args:
            crew_builder: The CrewAI builder instance
            grant_type: Type of grant
            topic: The grant topic

        Returns:
            Dict containing the enhanced tasks
        """
        if not hasattr(crew_builder, "create_tasks"):
            logger.error("Invalid crew_builder: missing create_tasks method")
            return {}

        try:
            # Get the original tasks
            original_tasks = crew_builder.create_tasks(grant_type, topic)

            # Enhance each task with hybrid prompts
            enhanced_tasks = {}
            for task_name, task in original_tasks.items():
                # Extract the section from the task description
                section = self._extract_section_from_task(task)

                if section:
                    # Create a hybrid prompt for this section
                    hybrid_result = self.create_hybrid_prompt(
                        grant_type=grant_type,
                        topic=topic,
                        section=section,
                        additional_context=f"This is for the {task_name} task in the grant writing process."
                    )

                    # Enhance the task description with the hybrid prompt
                    enhanced_description = hybrid_result["prompt"]

                    # Create a new task with the enhanced description
                    enhanced_task = task.copy()
                    enhanced_task.description = enhanced_description

                    enhanced_tasks[task_name] = enhanced_task
                    logger.info(
                        f"Enhanced task: {task_name} with hybrid prompt")
                else:
                    # Keep the original task if we couldn't extract a section
                    enhanced_tasks[task_name] = task
                    logger.warning(
                        f"Couldn't enhance task: {task_name} - using original")

            return enhanced_tasks

        except Exception as e:
            logger.error(f"Error enhancing CrewAI tasks: {e}")
            return {}

    def _extract_section_from_task(self, task) -> Optional[str]:
        """Extract the section name from a task description."""
        # Common section names in grant proposals
        common_sections = [
            "Executive Summary", "Problem Statement", "Background",
            "Methodology", "Expected Outcomes", "Budget", "Timeline",
            "Evaluation Plan", "Sustainability Plan", "Conclusion"
        ]

        # Check if any common section is mentioned in the task description
        description = task.description.lower()
        for section in common_sections:
            if section.lower() in description:
                return section

        return None

    def process_uploaded_file(
        self,
        file_obj,
        grant_type: str,
        custom_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process an uploaded file and add it to both samples and RAG.

        Args:
            file_obj: File-like object (from Streamlit or other source)
            grant_type: Type of grant
            custom_metadata: Additional metadata to store

        Returns:
            Dict with processing results
        """
        result = {
            "success": False,
            "added_to_samples": False,
            "added_to_rag": False,
            "error": None,
            "file_info": {
                "filename": getattr(file_obj, "name", "unknown"),
                "size": 0
            }
        }

        try:
            # Get file extension
            filename = getattr(file_obj, "name", "")
            file_ext = Path(filename).suffix.lower()

            # Check if file type is supported
            if not self.sample_processor:
                raise ValueError("Sample processor not initialized")

            if file_ext not in self.sample_processor.supported_extensions:
                raise ValueError(f"Unsupported file type: {file_ext}")

            # Create a temporary file to save the uploaded content
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
                temp_path = Path(temp_file.name)

                # Write the content to the temporary file
                file_obj.seek(0)
                temp_file.write(file_obj.read())
                temp_file.flush()

                # Get file size
                result["file_info"]["size"] = temp_path.stat().st_size

                # Create the target directory if it doesn't exist
                target_dir = self.sample_dir / grant_type
                target_dir.mkdir(parents=True, exist_ok=True)

                # Copy to the samples directory with a unique name
                timestamp = int(time.time())
                target_path = target_dir / f"user_upload_{timestamp}{file_ext}"
                target_path.write_bytes(temp_path.read_bytes())

                logger.info(f"Saved uploaded file to {target_path}")

                # Add to samples
                if self.config["use_samples"]:
                    # Extract text and metadata
                    content, metadata = self.sample_processor._extract_text_from_file(
                        target_path)

                    # Add custom metadata if provided
                    if custom_metadata:
                        metadata.update(custom_metadata)

                    # Validate and sanitize
                    content = self.sample_processor._validate_and_sanitize_text(
                        content)

                    if content:
                        # Make sure the grant type exists in samples
                        if grant_type not in self.sample_processor.samples:
                            self.sample_processor.samples[grant_type] = []

                        # Add to samples
                        self.sample_processor.samples[grant_type].append({
                            "filename": target_path.name,
                            "content": content,
                            "word_count": len(content.split()),
                            "sections": self.sample_processor._extract_sections(content),
                            "file_type": file_ext[1:],  # Remove the dot
                            "metadata": metadata
                        })

                        result["added_to_samples"] = True
                        logger.info(
                            f"Added {filename} to samples as {target_path.name}")

                # Add to RAG
                if self.config["use_rag"]:
                    if file_ext == ".pdf":
                        # Use the specialized PDF loader
                        load_and_store_pdf(
                            temp_path, f"User uploaded {grant_type} grant: {filename}")
                    else:
                        # Create a file-like object for other file types
                        with open(temp_path, 'r', encoding='utf-8') as f:
                            from utils.rag_utils import store_text_content
                            store_text_content(
                                f, f"User uploaded {grant_type} grant: {filename}")

                    result["added_to_rag"] = True
                    logger.info(f"Added {filename} to RAG system")

            # Clean up the temporary file
            os.unlink(temp_path)

            result["success"] = True

        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}")
            result["error"] = str(e)

        return result

    def get_integration_status(self) -> Dict[str, Any]:
        """Get the current status of the hybrid integration."""
        status = {
            "samples": {
                "available": self.sample_processor is not None,
                "count": sum(len(samples) for samples in self.samples.values()) if self.samples else 0,
                "grant_types": list(self.samples.keys()) if self.samples else [],
                "file_types": []
            },
            "rag": {
                "available": self.config["use_rag"] and Path(self.vector_db_dir).exists(),
                "vector_db_path": str(self.vector_db_dir)
            },
            "model": {
                "provider": self.model.split('/')[0] if self.model else None,
                "name": self.model.split('/')[1] if self.model and '/' in self.model else None
            },
            "config": self.config
        }

        # Get file type statistics if available
        if self.sample_processor:
            stats = self.sample_processor.get_sample_statistics()
            status["samples"]["file_types"] = list(
                stats.get("by_file_type", {}).keys())

            # Add detailed stats
            status["samples"]["statistics"] = stats

        return status

# Example usage functions


def example_standalone_usage():
    """Demonstrate standalone usage of the hybrid generator."""
    print("\n===== STANDALONE USAGE EXAMPLE =====")

    # Initialize the hybrid generator
    generator = HybridGrantGenerator(
        sample_dir="data/samples",
        config={
            "use_rag": True,
            "use_samples": True,
            "max_rag_results": 3
        }
    )

    # Add samples to RAG
    generator.add_samples_to_rag()

    # Create a hybrid prompt
    result = generator.create_hybrid_prompt(
        grant_type="research",
        topic="AI-powered healthcare diagnostics",
        section="Problem Statement"
    )

    # Print the result
    print("\nGenerated Hybrid Prompt:")
    print(f"Time taken: {result['generation_time']:.2f} seconds")

    if result["sample_used"]:
        print(
            f"Sample used: {result['sample_used']['filename']} ({result['sample_used']['file_type']})")

    if result["rag_results"]:
        print(f"RAG results: {len(result['rag_results'])} documents retrieved")

    print("\nPrompt excerpt:")
    print(result["prompt"][:500] + "...\n")

    # Print integration status
    status = generator.get_integration_status()
    print("\nIntegration Status:")
    print(
        f"Samples: {status['samples']['count']} samples across {len(status['samples']['grant_types'])} grant types")
    print(f"File types: {', '.join(status['samples']['file_types'])}")
    print(f"RAG available: {status['rag']['available']}")
    print(f"Model: {status['model']['provider']}/{status['model']['name']}")


def example_streamlit_integration():
    """
    Example code for integrating with Streamlit.

    Note: This is a code snippet, not meant to be executed directly.
    It shows how to integrate with the main Streamlit application.
    """
    print("\n===== STREAMLIT INTEGRATION EXAMPLE =====")
    print("The following is example code for Streamlit integration:")

    code_example = """
    # In your Streamlit app
    import streamlit as st
    from enhanced_sample_integration import HybridGrantGenerator
    
    # Initialize the generator (do this once)
    @st.cache_resource
    def get_hybrid_generator():
        return HybridGrantGenerator(
            sample_dir="data/samples",
            config={
                "use_rag": True,
                "use_samples": True,
                "max_rag_results": 3
            }
        )
    
    # Get the generator
    generator = get_hybrid_generator()
    
    # Add a file uploader for samples
    st.subheader("Upload Sample Grants")
    uploaded_file = st.file_uploader("Upload a sample grant document", 
                                    type=["pdf", "docx", "txt"])
    
    grant_type = st.selectbox("Grant Type", 
                             ["research", "nonprofit", "technology", "healthcare", "education"])
    
    if uploaded_file and st.button("Process Sample"):
        with st.spinner("Processing sample..."):
            result = generator.process_uploaded_file(
                uploaded_file,
                grant_type=grant_type,
                custom_metadata={"source": "user_upload", "timestamp": time.time()}
            )
            
            if result["success"]:
                st.success(f"Successfully processed {uploaded_file.name}")
                if result["added_to_samples"]:
                    st.info("Added to sample library")
                if result["added_to_rag"]:
                    st.info("Added to RAG system")
            else:
                st.error(f"Error: {result['error']}")
    
    # When generating a grant
    if st.button("Generate Grant"):
        topic = st.text_input("Grant Topic")
        section = st.selectbox("Section", ["Executive Summary", "Problem Statement", "Methodology"])
        
        with st.spinner("Generating..."):
            # Get the hybrid prompt
            prompt_result = generator.create_hybrid_prompt(
                grant_type=grant_type,
                topic=topic,
                section=section
            )
            
            # Use the prompt with your LLM
            # response = call_llm_with_prompt(prompt_result["prompt"])
            
            # Show details about the generation
            with st.expander("Generation Details"):
                st.write(f"Time: {prompt_result['generation_time']:.2f}s")
                if prompt_result["sample_used"]:
                    st.write(f"Sample: {prompt_result['sample_used']['filename']}")
                if prompt_result["rag_results"]:
                    st.write(f"RAG: {len(prompt_result['rag_results'])} documents")
    """

    print(code_example)


def example_crewai_integration():
    """
    Example code for integrating with CrewAI.

    Note: This is a code snippet, not meant to be executed directly.
    It shows how to integrate with the CrewAI workflow.
    """
    print("\n===== CREWAI INTEGRATION EXAMPLE =====")
    print("The following is example code for CrewAI integration:")

    code_example = """
    # In your CrewAI workflow
    from crew_agents.crew_builder import CrewBuilder
    from enhanced_sample_integration import HybridGrantGenerator
    
    # Initialize components
    crew_builder = CrewBuilder()
    hybrid_generator = HybridGrantGenerator()
    
    # Add samples to RAG
    hybrid_generator.add_samples_to_rag()
    
    # Create enhanced tasks
    enhanced_tasks = hybrid_generator.enhance_crew_tasks(
        crew_builder=crew_builder,
        grant_type="research",
        topic="AI-powered healthcare diagnostics"
    )
    
    # Use the enhanced tasks to create your crew
    agents = crew_builder.create_agents()
    
    # Create the crew with enhanced tasks
    crew = Crew(
        agents=list(agents.values()),
        tasks=list(enhanced_tasks.values()),
        verbose=True
    )
    
    # Run the crew
    result = crew.kickoff()
    """

    print(code_example)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced Sample Integration Example")
    parser.add_argument("--sample_dir", default="data/samples",
                        help="Directory containing grant samples")
    parser.add_argument("--grant_type", default="research",
                        help="Type of grant to generate")
    parser.add_argument(
        "--topic", default="AI-powered healthcare diagnostics", help="Grant topic")
    parser.add_argument(
        "--section", default="Problem Statement", help="Section to generate")
    parser.add_argument("--examples", action="store_true",
                        help="Show integration examples")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    print("Enhanced Sample Integration Example")
    print("==================================")

    if args.examples:
        # Show integration examples
        example_standalone_usage()
        example_streamlit_integration()
        example_crewai_integration()
    else:
        try:
            # Initialize the hybrid generator
            generator = HybridGrantGenerator(
                sample_dir=args.sample_dir,
                config={
                    "use_rag": True,
                    "use_samples": True,
                    "max_rag_results": 3
                }
            )

            # Add samples to RAG
            generator.add_samples_to_rag()

            # Get integration status
            status = generator.get_integration_status()
            print("\nIntegration Status:")
            print(
                f"Samples: {status['samples']['count']} samples across {len(status['samples']['grant_types'])} grant types")
            print(f"File types: {', '.join(status['samples']['file_types'])}")
            print(f"RAG available: {status['rag']['available']}")
            print(
                f"Model: {status['model']['provider']}/{status['model']['name']}")

            # Create a hybrid prompt
            print(
                f"\nGenerating hybrid prompt for {args.grant_type} grant about {args.topic}...")
            result = generator.create_hybrid_prompt(
                grant_type=args.grant_type,
                topic=args.topic,
                section=args.section
            )

            # Print the result
            print(f"\nTime taken: {result['generation_time']:.2f} seconds")

            if result["sample_used"]:
                print(
                    f"Sample used: {result['sample_used']['filename']} ({result['sample_used']['file_type']})")

            if result["rag_results"]:
                print(
                    f"RAG results: {len(result['rag_results'])} documents retrieved")

            print("\nPrompt excerpt:")
            print(result["prompt"][:500] + "...\n")

            print("To see integration examples, run with --examples flag")

        except Exception as e:
            logger.error(f"Error in example: {e}")
            print(f"\nError: {e}")
            print("\nTo see integration examples, run with --examples flag")
