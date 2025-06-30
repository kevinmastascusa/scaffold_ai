import json

try:
    with open('vector_outputs/scaffold_metadata_1.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total entries: {len(data)}")
    print(f"Keys in first entry: {list(data[0].keys())}")
    print(f"Has 'text' field: {'text' in data[0]}")
    
    if 'text' in data[0]:
        text_len = len(data[0]['text'])
        print(f"Text length in first entry: {text_len}")
        if text_len > 0:
            print(f"First 100 chars: {data[0]['text'][:100]}...")
        else:
            print("Text field is empty")
    else:
        print("No 'text' field found")
        
except Exception as e:
    print(f"Error: {e}") 