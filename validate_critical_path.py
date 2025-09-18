#!/usr/bin/env python3
"""
Critical Path Validation Script
Validates that all critical path components are present in the repository.
"""

import os
import json
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and return status."""
    path = Path(filepath)
    exists = path.exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists

def check_directory_exists(dirpath, description):
    """Check if a directory exists and return status."""
    path = Path(dirpath)
    exists = path.exists() and path.is_dir()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {dirpath}")
    return exists

def validate_critical_path():
    """Validate all critical path components."""
    print("🔍 Critical Path Component Validation")
    print("=" * 50)
    
    # Track overall status
    all_good = True
    
    print("\n📊 1. Data Preparation Pipeline")
    components = [
        ("scaffold_core/scripts/chunk/ChunkTest.py", "PDF Chunking Script"),
        ("scaffold_core/config.py", "Configuration Management"),
        ("outputs/", "Outputs Directory"),
    ]
    
    for filepath, desc in components:
        if not check_file_exists(filepath, desc) and not check_directory_exists(filepath, desc):
            all_good = False
    
    print("\n🧠 2. Vector Database & Embeddings")
    components = [
        ("scaffold_core/vector/main.py", "Vector Processing Main"),
        ("scaffold_core/vector/transformVector.py", "Vector Transformer"),
        ("scaffold_core/vector/enhanced_query.py", "Enhanced Query System"),
        ("vector_outputs/", "Vector Outputs Directory"),
    ]
    
    for filepath, desc in components:
        if not check_file_exists(filepath, desc) and not check_directory_exists(filepath, desc):
            all_good = False
    
    print("\n🤖 3. LLM Integration")
    components = [
        ("scaffold_core/llm.py", "LLM Integration Module"),
        ("generate_query_results.py", "Query Results Generator"),
        ("generate_query_responses.py", "Query Response Generator"),
    ]
    
    for filepath, desc in components:
        if not check_file_exists(filepath, desc):
            all_good = False
    
    print("\n🔗 4. Citation System")
    components = [
        ("scaffold_core/citation_handler.py", "Citation Handler"),
        ("test_citation_ui.py", "Citation UI Test"),
    ]
    
    for filepath, desc in components:
        if not check_file_exists(filepath, desc):
            all_good = False
    
    print("\n🎨 5. User Interface")
    components = [
        ("frontend/", "Frontend Directory"),
        ("frontend/app.py", "Flask Application"),
        ("frontend/templates/", "UI Templates Directory"),
    ]
    
    for filepath, desc in components:
        if not check_file_exists(filepath, desc) and not check_directory_exists(filepath, desc):
            all_good = False
    
    print("\n📋 6. Documentation & Testing")
    components = [
        ("CRITICAL_PATH.md", "Critical Path Documentation"),
        ("TEAM_ROLES.md", "Team Roles Documentation"),
        ("README.md", "Project README"),
        ("PROGRESS.md", "Progress Log"),
        ("documentation/", "Documentation Directory"),
    ]
    
    for filepath, desc in components:
        if not check_file_exists(filepath, desc) and not check_directory_exists(filepath, desc):
            all_good = False
    
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 All critical path components are present!")
        print("✅ Project structure validates successfully")
    else:
        print("⚠️  Some critical path components are missing")
        print("📝 Review the missing components above")
    
    return all_good

def generate_status_report():
    """Generate a status report for the critical path."""
    print("\n📊 Critical Path Status Report")
    print("=" * 50)
    
    status_map = {
        "Data Preparation": "✅ COMPLETED",
        "Vector Database": "✅ COMPLETED", 
        "LLM Integration": "✅ COMPLETED",
        "Citation System": "🔄 IN PROGRESS (80%)",
        "User Interface": "🔄 IN PROGRESS (75%)",
        "Quality Assurance": "📋 PLANNED",
        "Stakeholder Feedback": "📋 PLANNED",
        "Final Testing": "📋 PLANNED"
    }
    
    for component, status in status_map.items():
        print(f"{component:20} | {status}")
    
    print("\n🎯 Next Steps:")
    print("1. Complete citation integration in UI")
    print("2. Finalize UI testing and validation")
    print("3. Begin comprehensive system QA")
    
    return status_map

if __name__ == "__main__":
    print("🚀 Scaffold AI - Critical Path Validation")
    print("Date:", "July 20, 2025")
    print("Repository: kevinmastascusa/scaffold_ai")
    
    # Run validation
    validation_passed = validate_critical_path()
    
    # Generate status report
    status_report = generate_status_report()
    
    # Summary
    print(f"\n📈 Overall Project Status: 85% Complete")
    print(f"🎯 Target Completion: August 16, 2025")
    
    exit(0 if validation_passed else 1)