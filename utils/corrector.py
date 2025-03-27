import os
import json

# Open the broken JSON file
_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)),"LOGS")
with open(os.path.join(_dir,"Results.json"), "r") as file:
    json_text = file.read()

# Fix JSON by adding brackets and commas where necessary
fixed_json = "[" + json_text.replace("}{", "},{") + "]"

# Try parsing the fixed JSON
try:
    data = json.loads(fixed_json)
    print("JSON is now valid!")
    
    # Save the fixed JSON
    with open(os.path.join(_dir,"Results_fixed.json"), "w") as fixed_file:
        json.dump(data, fixed_file, indent=4)

except json.JSONDecodeError as e:
    print(f"Error: {e}")
