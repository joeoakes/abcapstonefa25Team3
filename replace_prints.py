import re
from pathlib import Path

# Path to your file
target = Path("QiskitShorsMain.py")

# Read the code
code = target.read_text(encoding="utf-8")

# Regex pattern to match print statements like:
# print("message") or print(f"message {var}")
pattern = re.compile(r'print\s*\((.*?)\)', re.DOTALL)

# Replace each print() with Log() keeping the same arguments
new_code = pattern.sub(r'Log(\1, Fore.WHITE)', code)

# Backup original file just in case
backup = target.with_suffix(".bak")
backup.write_text(code, encoding="utf-8")

# Write updated code
target.write_text(new_code, encoding="utf-8")

print("✅ All print() statements replaced with Log(..., Fore.WHITE)")
print(f"📁 Backup saved as: {backup}")
