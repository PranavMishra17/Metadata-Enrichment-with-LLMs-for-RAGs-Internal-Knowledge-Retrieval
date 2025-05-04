# add_gpu_to_all.py

import os
import re
import glob

# Directory of the git repository
REPO_DIR = "."  # Change this to your repository path if different

def modify_python_file(file_path):
    """Modify a Python file to use GPU via the GPUVerifier class."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if the file already imports our GPU utility
    if "from gpu_utils import GPUVerifier" in content or "import gpu_utils" in content:
        print(f"✓ File already has GPU imports: {file_path}")
        return False
    
    # Prepare import statement and initialization code
    gpu_import = "from gpu_utils import GPUVerifier\n"
    gpu_init = "\n# Initialize GPU verification\ngpu_verifier = GPUVerifier(require_gpu=True)\n"
    
    # Find appropriate position to insert the import
    import_match = re.search(r"^(import .*|from .* import .*)\n", content, re.MULTILINE)
    if import_match:
        # Add after existing imports
        last_import_pos = 0
        for match in re.finditer(r"^(import .*|from .* import .*)\n", content, re.MULTILINE):
            last_import_pos = match.end()
        
        new_content = content[:last_import_pos] + "\n" + gpu_import + content[last_import_pos:]
    else:
        # No imports found, add at the beginning
        new_content = gpu_import + content
    
    # Add GPU initialization after docstring or at the beginning of the main code
    main_block_match = re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", new_content)
    if main_block_match:
        # Add before the main block
        main_pos = main_block_match.start()
        new_content = new_content[:main_pos] + gpu_init + new_content[main_pos:]
    else:
        # Try to find after imports and docstrings
        docstring_pattern = r'""".*?"""'
        last_pos = 0
        
        # Find the end of multiline docstrings
        docstring_match = re.search(docstring_pattern, new_content, re.DOTALL)
        if docstring_match:
            last_pos = docstring_match.end()
        
        # If no docstring, add after imports or at the beginning
        if last_pos == 0:
            # Find the last import
            for match in re.finditer(r"^(import .*|from .* import .*)\n", new_content, re.MULTILINE):
                last_pos = match.end()
        
        # If still no position found, add after any potential comments at the beginning
        if last_pos == 0:
            comment_match = re.match(r"^(#.*\n)+", new_content)
            if comment_match:
                last_pos = comment_match.end()
                
        # Add initialization code
        new_content = new_content[:last_pos] + gpu_init + new_content[last_pos:]
    
    # Write the modified content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ Added GPU verification to: {file_path}")
    return True

def process_repository():
    """Process all Python files in the repository."""
    # Make sure gpu_utils.py exists
    if not os.path.exists(os.path.join(REPO_DIR, "gpu_utils.py")):
        with open(os.path.join(REPO_DIR, "gpu_utils.py"), 'w', encoding='utf-8') as f:
            # Copy the content of the GPUVerifier class here
            print("Creating gpu_utils.py...")
            # Get the content of the GPUVerifier class from the first code block in this file
            with open(__file__, 'r', encoding='utf-8') as this_file:
                content = this_file.read()
                gpu_utils_content = re.search(r"```python\n# gpu_utils\.py\n\n(.*?)\n```", content, re.DOTALL)
                if gpu_utils_content:
                    f.write(gpu_utils_content.group(1))
                else:
                    print("Error: Couldn't extract GPUVerifier class content.")
                    return
    
    # Find all Python files
    python_files = glob.glob(os.path.join(REPO_DIR, "**", "*.py"), recursive=True)
    
    # Skip the utility files themselves
    python_files = [f for f in python_files if os.path.basename(f) not in ["gpu_utils.py", "add_gpu_to_all.py"]]
    
    modified_count = 0
    for file_path in python_files:
        try:
            if modify_python_file(file_path):
                modified_count += 1
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    print(f"\nSummary: Modified {modified_count} of {len(python_files)} Python files.")
    print("To verify GPU usage:")
    print("1. Run any of your scripts normally")
    print("2. During execution, open another terminal and run 'nvidia-smi' to check GPU usage")
    print("\nYou may need to adjust the GPU requirements in individual files if needed.")

if __name__ == "__main__":
    process_repository()