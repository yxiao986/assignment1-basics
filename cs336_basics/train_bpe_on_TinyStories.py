import wandb
import time
import json
from .train_bpe import train_bpe

input_path = "data/TinyStoriesV2-GPT4-train.txt"
vocab_size = 10000
special_tokens = ["<|endoftext|>"]
vocab_out_path = "data/vocab_TinyStories.txt"
merges_out_path = "data/merges_TinyStories.txt"

start_time = time.time()
print(f"Start training BPE tokenizer on TinyStories.")

vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

end_time = time.time()
print(f"Training finished in {end_time - start_time:.2f} seconds.")

serializable_vocab = {
        k: v.decode('utf-8', errors='backslashreplace') for k, v in vocab.items()
    }

with open(vocab_out_path, 'w', encoding='utf-8') as f:
    json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)

with open(merges_out_path, 'w', encoding='utf-8') as f:
        for pair in merges:
            
            p1_str = pair[0].decode('utf-8', errors='backslashreplace')
            p2_str = pair[1].decode('utf-8', errors='backslashreplace')
            f.write(f"{p1_str} {p2_str}\n")