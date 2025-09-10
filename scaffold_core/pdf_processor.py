"""
PDF Processing Module for Scaffold AI
Handles syllabus PDF uploads and content extraction.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logger.warning("PyPDF2 not available - PDF processing will be limited")

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    logger.warning("pypdf not available - PDF processing will be limited")


class SyllabusProcessor:
    """Process uploaded syllabus PDFs and extract relevant content."""
    
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text content from PDF file."""
        try:
            text_content = ""
            
            if PYPDF_AVAILABLE:
                # Use pypdf (newer library)
                with open(pdf_path, 'rb') as file:
                    reader = PdfReader(file)
                    for page in reader.pages:
                        text_content += page.extract_text() + "\n"
                        
            elif PYPDF2_AVAILABLE:
                # Use PyPDF2 (older library)
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text_content += page.extract_text() + "\n"
            else:
                raise ImportError("No PDF processing library available")
            # Normalize at source to reduce OCR artifacts
            try:
                from scaffold_core.text_clean import normalize_extracted_text
                text_content = normalize_extracted_text(text_content)
            except Exception:
                text_content = text_content.strip()

            return text_content
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            raise
    
    def analyze_syllabus_content(self, text_content: str) -> Dict[str, Any]:
        """Analyze syllabus content and extract key information."""
        analysis = {
            'course_info': {},
            'topics': [],
            'learning_objectives': [],
            'assessment_methods': [],
            'sustainability_opportunities': []
        }
        
        # Extract course information
        lines = text_content.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Course title
            if any(keyword in line_lower for keyword in ['course title', 'course name', 'title:']):
                if i + 1 < len(lines):
                    analysis['course_info']['title'] = lines[i + 1].strip()
            
            # Course code
            if any(keyword in line_lower for keyword in ['course code', 'course number', 'code:']):
                if i + 1 < len(lines):
                    analysis['course_info']['code'] = lines[i + 1].strip()
            
            # Topics/Content
            if any(keyword in line_lower for keyword in ['topics', 'content', 'syllabus', 'course outline']):
                # Look for bullet points or numbered items
                j = i + 1
                while j < len(lines) and j < i + 20:  # Look ahead 20 lines
                    line_content = lines[j].strip()
                    if line_content and not line_content.lower().startswith(('course', 'instructor', 'office')):
                        if any(char in line_content for char in ['•', '-', '*', '1.', '2.', '3.']):
                            analysis['topics'].append(line_content)
                    j += 1
            
            # Learning objectives
            if any(keyword in line_lower for keyword in ['learning objectives', 'objectives', 'outcomes', 'goals']):
                j = i + 1
                while j < len(lines) and j < i + 15:
                    line_content = lines[j].strip()
                    if line_content and not line_content.lower().startswith(('course', 'instructor', 'office')):
                        if any(char in line_content for char in ['•', '-', '*', '1.', '2.', '3.']):
                            analysis['learning_objectives'].append(line_content)
                    j += 1
        
        # Identify sustainability opportunities
        sustainability_keywords = [
            'environment', 'environmental', 'sustainability', 'sustainable', 'green',
            'energy', 'renewable', 'climate', 'carbon', 'waste', 'recycling',
            'conservation', 'efficiency', 'life cycle', 'impact', 'footprint'
        ]
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in sustainability_keywords):
                analysis['sustainability_opportunities'].append(line.strip())
        
        return analysis
    
    def generate_sustainability_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate sustainability integration suggestions based on syllabus analysis."""
        suggestions = []
        
        course_title = analysis['course_info'].get('title', '').lower()
        topics = analysis['topics']
        objectives = analysis['learning_objectives']
        
        # General suggestions based on course type
        if any(word in course_title for word in ['mechanics', 'dynamics', 'statics']):
            suggestions.extend([
                "Incorporate energy efficiency principles in mechanical systems design",
                "Add life cycle assessment for mechanical components",
                "Include sustainable materials selection in design projects",
                "Discuss renewable energy applications in mechanical systems"
            ])
        
        if any(word in course_title for word in ['thermodynamics', 'heat', 'energy']):
            suggestions.extend([
                "Integrate renewable energy systems analysis",
                "Add energy conservation principles and applications",
                "Include sustainable energy conversion technologies",
                "Discuss carbon footprint reduction in energy systems"
            ])
        
        if any(word in course_title for word in ['fluid', 'hydraulics', 'pneumatics']):
            suggestions.extend([
                "Add sustainable fluid system design principles",
                "Include water conservation and efficiency topics",
                "Discuss renewable energy applications in fluid systems",
                "Integrate environmental impact assessment of fluid systems"
            ])
        
        if any(word in course_title for word in ['materials', 'properties', 'structure']):
            suggestions.extend([
                "Include sustainable materials selection criteria",
                "Add life cycle assessment of materials",
                "Discuss recycling and circular economy principles",
                "Integrate environmental impact of material choices"
            ])
        
        # Specific suggestions based on existing topics
        for topic in topics:
            topic_lower = topic.lower()
            
            if any(word in topic_lower for word in ['design', 'project']):
                suggestions.append(f"Modify '{topic}' to include sustainability criteria and environmental impact assessment")
            
            if any(word in topic_lower for word in ['analysis', 'calculation']):
                suggestions.append(f"Add sustainability metrics and environmental impact calculations to '{topic}'")
            
            if any(word in topic_lower for word in ['case study', 'example']):
                suggestions.append(f"Include sustainability-focused case studies in '{topic}'")
        
        return list(set(suggestions))  # Remove duplicates
    
    def process_uploaded_syllabus(self, file_path: Path, session_id: str) -> Dict[str, Any]:
        """Process an uploaded syllabus PDF and return analysis results."""
        try:
            # Extract text from PDF
            text_content = self.extract_text_from_pdf(file_path)
            
            # Analyze content
            analysis = self.analyze_syllabus_content(text_content)
            
            # Generate suggestions
            suggestions = self.generate_sustainability_suggestions(analysis)
            
            # Create results
            results = {
                'session_id': session_id,
                'filename': file_path.name,
                'file_path': str(file_path),
                'text_content': text_content[:1000] + "..." if len(text_content) > 1000 else text_content,
                'analysis': analysis,
                'sustainability_suggestions': suggestions,
                'processing_status': 'success'
            }
            
            # Save results
            results_file = self.upload_dir / f"{session_id}_analysis.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Successfully processed syllabus: {file_path.name}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing syllabus {file_path}: {e}")
            return {
                'session_id': session_id,
                'filename': file_path.name,
                'file_path': str(file_path),
                'processing_status': 'error',
                'error_message': str(e)
            }


# Global instance
syllabus_processor = SyllabusProcessor()


def process_syllabus_upload(file_path: str, session_id: str) -> Dict[str, Any]:
    """Convenience function to process syllabus uploads."""
    return syllabus_processor.process_uploaded_syllabus(Path(file_path), session_id) 