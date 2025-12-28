
import json

nb_path = "/root/NeMo/verify_persian_tts.ipynb"

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Iterate through cells to find the tokenizer init cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        source_text = "".join(source)
        if "PersianPhonemesTokenizer" in source_text and "use_pause_tokens" in source_text:
            print("Found target cell, updating...")
            new_source = [
                "tokenizer = PersianPhonemesTokenizer(\n",
                "    g2p=g2p,\n",
                "    punct=True\n",
                ")\n",
                "print(\"✅ Tokenizer initialized.\")\n",
                "\n",
                "# Check vocabulary size\n",
                "print(f\"Tokenizer Vocab Size: {len(tokenizer.tokens)}\")"
            ]
            cell['source'] = new_source
            break
else:
    print("Warning: Target cell not found!")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False) # indent changed to 4 usually standard, original looked like 4? actually original was 1? or 4?
    # Viewer showed 4 spaces indentation in JSON structure.
    # nbformat usually uses 1 space or 2?
    # I'll stick to indent=1 or just let json dump do its thing.
    # Actually, let's check original file indentation style if possible from view_file output.
    # It seems to be 4 spaces.

print("Notebook updated.")
