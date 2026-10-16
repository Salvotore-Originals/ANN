import platform, os, sys
import numpy, faiss, psutil

import subprocess
name = subprocess.run(["powershell","-NoProfile","-Command","(Get-CimInstance Win32_Processor).Name"],
                      capture_output=True, text=True).stdout.strip()

out = [
    f"OS: {platform.platform()}",
    f"CPU: {platform.processor()}",
    f"Logical cores: {os.cpu_count()}",
    f"RAM GB: {psutil.virtual_memory().total/2**30:.1f}",
    f"Python: {sys.version.split()[0]}",
    f"numpy: {numpy.__version__}",
    f"CPU name: {name}",
    f"faiss: {faiss.__version__}",
]
os.makedirs("../results", exist_ok=True)
open("../results/env.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))