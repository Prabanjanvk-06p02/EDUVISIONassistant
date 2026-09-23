import os
import json
from pathlib import Path

brain_dir = Path(r"C:\Users\prabanjan v\.gemini\antigravity-ide\brain")

found_any = False
for conv_dir in brain_dir.iterdir():
    if not conv_dir.is_dir(): continue
    
    transcript_path = conv_dir / ".system_generated" / "logs" / "transcript.jsonl"
    if not transcript_path.exists(): continue
        
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        messages = []
        is_match = False
        
        for line in lines:
            try:
                data = json.loads(line)
            except:
                continue
            
            # Extract user messages
            if data.get('source') == 'USER_EXPLICIT' and data.get('type') == 'USER_INPUT':
                content = data.get('content', '')
                req_start = content.find('<USER_REQUEST>')
                req_end = content.find('</USER_REQUEST>')
                if req_start != -1 and req_end != -1:
                    user_text = content[req_start + 14:req_end].strip()
                else:
                    user_text = content
                    
                time_str = data.get('created_at', '')
                messages.append(f"[USER] ({time_str}):\n{user_text}\n")
                
                # Check if this looks like the "hi" conversation
                # The user mentioned 6:00 PM (which would be ~12:30 UTC for IST timezone)
                if user_text.lower() == 'hi':
                    is_match = True
            
            # Extract agent responses
            elif data.get('source') == 'MODEL' and 'content' in data and data['content']:
                text = data['content']
                messages.append(f"[AGENT]:\n{text}\n")
                
        if is_match:
            print(f"\n======================================")
            print(f"CONVERSATION ID: {conv_dir.name}")
            print(f"======================================")
            for m in messages:
                print(m)
            found_any = True
                
    except Exception as e:
        continue

if not found_any:
    print("Could not find a conversation where you just said 'hi'.")
