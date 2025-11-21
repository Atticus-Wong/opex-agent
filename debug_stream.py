import asyncio
import json
import httpx
import time

async def main():
    url = "http://127.0.0.1:3001/chat"
    payload = {
        "chat_session_id": "debug-pretty", 
        "prompt": "Draft a workflow for onboarding new designers."
    }
    
    print(f"--- Connecting to {url} ---")
    start_time = time.time()
    
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload) as resp:
            print(f"Connected! Status Code: {resp.status_code}\n")
            
            async for line in resp.aiter_lines():
                # Skip empty keep-alive lines
                if not line.strip():
                    continue
                    
                if line.startswith("data: "):
                    # Strip the 'data: ' prefix
                    raw_data = line[6:].strip()
                    
                    # Calculate time elapsed since request started
                    elapsed = time.time() - start_time
                    
                    if raw_data == "[DONE]":
                        print(f"[{elapsed:.3f}s] ✅ Stream finished.")
                        break
                    
                    try:
                        data = json.loads(raw_data)
                        # Pretty print the JSON object
                        print(f"[{elapsed:.3f}s] 📦 Event Received:")
                        print(json.dumps(data, indent=2))
                        print("-" * 40)
                    except json.JSONDecodeError:
                        print(f"[{elapsed:.3f}s] ⚠️ Parse Error: {raw_data}")

if __name__ == "__main__":
    # If running in a standard script:
    asyncio.run(main())
    
    # If running in Jupyter/IPython, comment out the line above 
    # and uncomment the line below:
    # await main()
