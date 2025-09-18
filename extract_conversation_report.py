#!/usr/bin/env python3
"""
Script to extract conversation details from Scaffold AI conversation JSON file
and format into a comprehensive text report for professors.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

def extract_conversation_report(json_file_path, output_file_path):
    """
    Extract conversation details and create a formatted report.
    
    Args:
        json_file_path (str): Path to the conversation JSON file
        output_file_path (str): Path for the output text file
    """
    
    try:
        # Load the conversation JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            conversation_data = json.load(f)
        
        # Initialize the report
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("SCAFFOLD AI CONVERSATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Source File: {json_file_path}")
        report_lines.append("")
        
        # Extract syllabus information
        syllabus_info = None
        for item in conversation_data:
            if item.get('type') == 'syllabus_context':
                syllabus_info = item
                break
        
        if syllabus_info:
            report_lines.append("SYLLABUS INFORMATION")
            report_lines.append("-" * 40)
            report_lines.append(f"Course: {syllabus_info.get('content', '').split('Course: ')[1].split('\n')[0] if 'Course: ' in syllabus_info.get('content', '') else 'Unknown'}")
            report_lines.append(f"Filename: {syllabus_info.get('filename', 'Unknown')}")
            report_lines.append(f"Upload Time: {syllabus_info.get('timestamp', 'Unknown')}")
            report_lines.append("")
        
        # Extract Q&A pairs
        qa_pairs = []
        for item in conversation_data:
            if item.get('type') in ['user', 'assistant']:
                qa_pairs.append(item)
        
        # Group Q&A pairs
        conversations = []
        current_conversation = []
        
        for item in qa_pairs:
            if item['type'] == 'user':
                if current_conversation:
                    conversations.append(current_conversation)
                current_conversation = [item]
            elif item['type'] == 'assistant':
                current_conversation.append(item)
        
        if current_conversation:
            conversations.append(current_conversation)
        
        # Write conversation details
        report_lines.append("CONVERSATION DETAILS")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Questions: {len([item for item in qa_pairs if item['type'] == 'user'])}")
        report_lines.append(f"Total Responses: {len([item for item in qa_pairs if item['type'] == 'assistant'])}")
        report_lines.append("")
        
        # Write each Q&A pair
        for i, conversation in enumerate(conversations, 1):
            if len(conversation) >= 2:
                user_msg = conversation[0]
                assistant_msg = conversation[1]
                
                report_lines.append(f"QUESTION {i}")
                report_lines.append("=" * 20)
                report_lines.append(f"Time: {user_msg.get('timestamp', 'Unknown')}")
                report_lines.append(f"Question: {user_msg.get('content', '')}")
                report_lines.append("")
                
                report_lines.append("RESPONSE")
                report_lines.append("-" * 10)
                response_content = assistant_msg.get('content', '')
                # Clean up response formatting
                response_content = response_content.replace('\n\n', '\n').strip()
                report_lines.append(response_content)
                report_lines.append("")
                
                # Add sources if available
                sources = assistant_msg.get('sources', [])
                if sources:
                    report_lines.append("SOURCES")
                    report_lines.append("-" * 8)
                    for j, source in enumerate(sources[:5], 1):  # Limit to top 5 sources
                        source_info = source.get('source', {})
                        score = source.get('score', 0)
                        preview = source.get('text_preview', '')[:200] + "..." if len(source.get('text_preview', '')) > 200 else source.get('text_preview', '')
                        
                        report_lines.append(f"{j}. {source_info.get('name', 'Unknown')}")
                        report_lines.append(f"   Relevance Score: {score:.3f}")
                        report_lines.append(f"   Preview: {preview}")
                        report_lines.append("")
                
                report_lines.append("=" * 80)
                report_lines.append("")
        
        # Summary statistics
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-" * 40)
        
        # Calculate response lengths
        response_lengths = []
        for item in qa_pairs:
            if item['type'] == 'assistant':
                content = item.get('content', '')
                response_lengths.append(len(content.split()))
        
        if response_lengths:
            avg_length = sum(response_lengths) / len(response_lengths)
            report_lines.append(f"Average Response Length: {avg_length:.1f} words")
            report_lines.append(f"Shortest Response: {min(response_lengths)} words")
            report_lines.append(f"Longest Response: {max(response_lengths)} words")
        
        # Source statistics
        total_sources = 0
        unique_sources = set()
        for item in qa_pairs:
            if item['type'] == 'assistant':
                sources = item.get('sources', [])
                total_sources += len(sources)
                for source in sources:
                    source_info = source.get('source', {})
                    unique_sources.add(source_info.get('name', 'Unknown'))
        
        report_lines.append(f"Total Sources Cited: {total_sources}")
        report_lines.append(f"Unique Sources: {len(unique_sources)}")
        report_lines.append("")
        
        # Footer
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        # Write the report to file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ Report successfully generated: {output_file_path}")
        print(f"📊 Processed {len(qa_pairs)} conversation items")
        print(f"📝 Report contains {len(report_lines)} lines")
        
    except FileNotFoundError:
        print(f"❌ Error: File not found: {json_file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON file: {json_file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

def main():
    """Main function to run the script."""
    
    # Default conversation file path
    default_conversation_file = "conversations/daf06e3c-342a-443e-a743-6f006f3b363b.json"
    
    # Check if conversation file exists
    if not Path(default_conversation_file).exists():
        print("❌ Error: Conversation file not found.")
        print(f"Expected file: {default_conversation_file}")
        print("\nAvailable conversation files:")
        
        # List available conversation files
        conversation_dirs = ["conversations", "frontend/conversations"]
        for conv_dir in conversation_dirs:
            if Path(conv_dir).exists():
                json_files = list(Path(conv_dir).glob("*.json"))
                for json_file in json_files:
                    size = json_file.stat().st_size
                    print(f"  {json_file} ({size} bytes)")
        
        sys.exit(1)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"conversation_report_{timestamp}.txt"
    
    print("🔍 Extracting conversation details...")
    print(f"📁 Input file: {default_conversation_file}")
    print(f"📄 Output file: {output_file}")
    print()
    
    # Extract the report
    extract_conversation_report(default_conversation_file, output_file)
    
    print()
    print("🎉 Report generation complete!")
    print(f"📋 You can now share '{output_file}' with your professor.")

if __name__ == "__main__":
    main() 