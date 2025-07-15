#!/usr/bin/env python3
"""
Test script to verify the citation layer and UI functionality.
"""

import os
import sys
import json

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)


def test_citation_handler():
    """Test the citation handler functionality."""
    print("🧪 Testing Citation Handler...")
    
    try:
        from scaffold_core.citation_handler import Citation
        
        # Test with a sample file path
        test_path = ("data/Climate Pedagogy Incubator/"
                    "Key competencies in sustainability a reference "
                    "framework for academic program development.pdf")
        
        citation = Citation(test_path)
        print("✅ Citation created successfully:")
        print(f"   - ID: {citation.id}")
        print(f"   - Name: {citation.clean_name}")
        print(f"   - Raw Path: {citation.raw_path}")
        
        # Test to_dict method
        citation_dict = citation.to_dict()
        print(f"✅ Citation dictionary: {json.dumps(citation_dict, indent=2)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Citation handler test failed: {e}")
        return False


def test_enhanced_query_system():
    """Test the enhanced query system with citation integration."""
    print("\n🧪 Testing Enhanced Query System...")
    
    try:
        from scaffold_core.vector.enhanced_query import query_enhanced
        
        # Test query
        test_query = "What is life cycle assessment?"
        print(f"📝 Testing query: '{test_query}'")
        
        result = query_enhanced(test_query)
        
        print("✅ Query completed successfully!")
        print(f"   - Response length: {len(result.get('response', ''))}")
        print(f"   - Candidates found: "
              f"{result.get('search_stats', {}).get('final_candidates', 0)}")
        
        # Check if citations are included
        candidates = result.get('candidates', [])
        if candidates:
            first_source = candidates[0].get('source', {}).get('name', 'Unknown')
            first_id = candidates[0].get('source', {}).get('id', 'Unknown')
            print(f"   - First candidate source: {first_source}")
            print(f"   - Citation ID: {first_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced query system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ui_api_response():
    """Test the UI API response format."""
    print("\n🧪 Testing UI API Response Format...")
    
    try:
        from scaffold_core.vector.enhanced_query import query_enhanced
        
        test_query = "What are sustainability competencies?"
        result = query_enhanced(test_query)
        
        # Format response as the UI would expect
        ui_response = {
            'query': test_query,
            'response': result['response'],
            'candidates_found': result['search_stats']['final_candidates'],
            'search_stats': result['search_stats'],
            'sources': [
                {
                    'source': candidate.get('source', {}).get('name', 'Unknown'),
                    'score': candidate.get('cross_score', candidate.get('score', 0)),
                    'text_preview': (
                        candidate.get('text', '')[:200] + '...'
                        if len(candidate.get('text', '')) > 200
                        else candidate.get('text', '')
                    )
                }
                for candidate in result['candidates'][:5]
            ]
        }
        
        print(f"✅ UI API response formatted successfully!")
        print(f"   - Sources count: {len(ui_response['sources'])}")
        print(f"   - First source: {ui_response['sources'][0]['source']}")
        print(f"   - First source score: {ui_response['sources'][0]['score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ UI API response test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Scaffold AI UI and Citation Layer")
    print("=" * 50)
    
    tests = [
        test_citation_handler,
        test_enhanced_query_system,
        test_ui_api_response
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The UI and citation layer are working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main() 