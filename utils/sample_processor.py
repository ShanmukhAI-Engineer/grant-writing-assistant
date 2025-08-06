"""
Sample-Based Grant Generation

This script processes your grant samples and uses them to generate new grants
that match the style, structure, and approach of successful proposals.
"""

import os
import glob
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SampleGrantProcessor:
    """Process grant samples to extract style and structure for generation."""
    
    def __init__(self, samples_dir: str = "data/samples"):
        self.samples_dir = Path(samples_dir)
        self.grant_types = ["research", "nonprofit", "technology", "healthcare", "education"]
        self.samples = {}
        
    def load_samples(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load all grant samples from the samples directory."""
        samples = {}
        
        for grant_type in self.grant_types:
            type_dir = self.samples_dir / grant_type
            if type_dir.exists():
                samples[grant_type] = []
                
                # Load all text files in the directory
                for file_path in type_dir.glob("*.txt"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        samples[grant_type].append({
                            "filename": file_path.name,
                            "content": content,
                            "word_count": len(content.split()),
                            "sections": self._extract_sections(content)
                        })
                        logger.info(f"Loaded sample: {file_path.name}")
                        
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")
        
        self.samples = samples
        return samples
    
    def _extract_sections(self, content: str) -> List[str]:
        """Extract section headings from the grant content."""
        lines = content.split('\n')
        sections = []
        
        for line in lines:
            line = line.strip()
            # Common grant section patterns
            if (line.isupper() or 
                line.startswith('##') or 
                line.startswith('TITLE:') or
                line.startswith('EXECUTIVE SUMMARY') or
                line.startswith('PROBLEM STATEMENT') or
                line.startswith('METHODOLOGY') or
                line.startswith('EXPECTED OUTCOMES') or
                line.startswith('BUDGET') or
                'SUMMARY' in line.upper() or
                'STATEMENT' in line.upper() or
                'METHODOLOGY' in line.upper() or
                'OUTCOMES' in line.upper()):
                sections.append(line)
        
        return sections
    
    def get_best_sample(self, grant_type: str, topic: str = None) -> Optional[Dict[str, Any]]:
        """Get the best matching sample for a given grant type and topic."""
        if grant_type not in self.samples or not self.samples[grant_type]:
            # Fallback to any available sample
            for gtype, samples_list in self.samples.items():
                if samples_list:
                    logger.info(f"Using {gtype} sample as fallback for {grant_type}")
                    return samples_list[0]
            return None
        
        # For now, return the first sample of the requested type
        # TODO: Add topic similarity matching
        return self.samples[grant_type][0]
    
    def create_style_prompt(self, sample: Dict[str, Any], user_topic: str, section: str) -> str:
        """Create a prompt that incorporates the sample's style and structure."""
        if not sample:
            return f"Write a {section} for a grant proposal about {user_topic}."
        
        # Extract relevant sections from the sample
        content = sample["content"]
        sections = sample["sections"]
        
        # Find the matching section in the sample
        sample_section = ""
        for sec in sections:
            if section.upper() in sec.upper():
                # Extract the content under this section
                start_idx = content.find(sec)
                if start_idx != -1:
                    # Find the next section or end of content
                    remaining_content = content[start_idx + len(sec):]
                    next_section_idx = float('inf')
                    
                    for other_sec in sections:
                        if other_sec != sec:
                            idx = remaining_content.find(other_sec)
                            if idx != -1 and idx < next_section_idx:
                                next_section_idx = idx
                    
                    if next_section_idx == float('inf'):
                        sample_section = remaining_content
                    else:
                        sample_section = remaining_content[:next_section_idx]
                    
                    sample_section = sample_section.strip()
                    break
        
        # Create the style-aware prompt
        if sample_section:
            prompt = f"""
            Write a {section} for a grant proposal about "{user_topic}".
            
            Use this successful example as a style and structure guide:
            
            EXAMPLE {section.upper()}:
            {sample_section[:800]}...
            
            Match the writing style, tone, and structure of the example while adapting the content 
            to focus on "{user_topic}". Maintain the same level of detail and professional language.
            """
        else:
            # Fallback to general style guidance
            first_500_words = " ".join(content.split()[:500])
            prompt = f"""
            Write a {section} for a grant proposal about "{user_topic}".
            
            Use this successful grant proposal as a style guide:
            
            STYLE EXAMPLE:
            {first_500_words}...
            
            Match the writing style, tone, and level of formality while focusing on "{user_topic}".
            """
        
        return prompt
    
    def get_sample_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded samples."""
        stats = {
            "total_samples": sum(len(samples) for samples in self.samples.values()),
            "by_type": {},
            "average_length": 0
        }
        
        total_words = 0
        total_samples = 0
        
        for grant_type, samples_list in self.samples.items():
            if samples_list:
                stats["by_type"][grant_type] = {
                    "count": len(samples_list),
                    "avg_words": sum(s["word_count"] for s in samples_list) / len(samples_list),
                    "sections": []
                }
                
                # Collect all sections
                all_sections = []
                for sample in samples_list:
                    all_sections.extend(sample["sections"])
                stats["by_type"][grant_type]["sections"] = list(set(all_sections))
                
                total_words += sum(s["word_count"] for s in samples_list)
                total_samples += len(samples_list)
        
        if total_samples > 0:
            stats["average_length"] = total_words / total_samples
        
        return stats
    
    def add_sample_to_rag(self, rag_utils):
        """Add all samples to the RAG system for enhanced context retrieval."""
        try:
            from utils.rag_utils import store_text_content
            
            for grant_type, samples_list in self.samples.items():
                for sample in samples_list:
                    # Create a temporary file-like object
                    import io
                    text_file = io.StringIO(sample["content"])
                    text_file.name = f"{grant_type}_{sample['filename']}"
                    
                    # Store in RAG system
                    store_text_content(text_file, f"Sample {grant_type} grant: {sample['filename']}")
                    logger.info(f"Added {sample['filename']} to RAG system")
                    
        except Exception as e:
            logger.error(f"Error adding samples to RAG: {e}")

# Usage functions
def process_user_samples():
    """Process user samples and return the processor."""
    processor = SampleGrantProcessor()
    samples = processor.load_samples()
    
    print(f"📋 Loaded {sum(len(s) for s in samples.values())} grant samples")
    
    stats = processor.get_sample_statistics()
    print("\n📊 Sample Statistics:")
    for grant_type, info in stats["by_type"].items():
        print(f"  {grant_type}: {info['count']} samples, avg {info['avg_words']:.0f} words")
    
    return processor

def generate_with_sample_style(processor, grant_type: str, topic: str, section: str):
    """Generate a proposal section using sample style."""
    sample = processor.get_best_sample(grant_type, topic)
    
    if sample:
        print(f"✅ Using sample: {sample['filename']} for {grant_type} grant")
        style_prompt = processor.create_style_prompt(sample, topic, section)
        return style_prompt
    else:
        print(f"❌ No samples found for {grant_type}")
        return f"Write a {section} for a grant proposal about {topic}."

if __name__ == "__main__":
    # Demo usage
    processor = process_user_samples()
    
    # Example: Generate a problem statement using research grant style
    prompt = generate_with_sample_style(
        processor, 
        "research", 
        "AI-powered healthcare diagnostics", 
        "Problem Statement"
    )
    
    print("\n📝 Generated Style-Aware Prompt:")
    print(prompt)
