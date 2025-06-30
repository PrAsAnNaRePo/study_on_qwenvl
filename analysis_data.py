#!/usr/bin/env python3
"""
Compute and print token count statistics for conversations in JSON file using Qwen tokenizer.
"""
import argparse
import json
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Analyze token count statistics for conversations in JSON file")
    parser.add_argument("input_file", help="Path to the input JSON file")
    args = parser.parse_args()
    
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{args.input_file}' not found.")
        return
    # Load Qwen tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to load Qwen tokenizer: {e}")
        return
    token_counts = []
    for item in data:
        # Build conversation string with special markers
        text = ""
        for msg in item.get("conversations", []):
            role = msg.get("from", "")
            content = msg.get("value", "").strip()
            text += f"<|im_start|>{role} {content}<|im_end|>\n"
        # Tokenize without adding extra special tokens
        enc = tokenizer(text, add_special_tokens=False)
        token_counts.append(len(enc.input_ids))
    if not token_counts:
        print("No conversation data to tokenize.")
        return
    # Print token count statistics
    print(f"Maximum tokens: {max(token_counts)}")
    print(f"Average tokens: {sum(token_counts) / len(token_counts):.2f}")
    min_tokens = min(token_counts)
    print(f"Minimum tokens: {min_tokens}")

    # Plot distribution of token counts
    plt.figure(figsize=(10, 6))
    plt.hist(token_counts, bins=50, color='skyblue', edgecolor='black')
    plt.title('Token Count Distribution')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    # Save plot to file
    plt.savefig('token_distribution.png')
    # Display plot
    plt.show()

if __name__ == "__main__":
    main()