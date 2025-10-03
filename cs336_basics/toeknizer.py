from typing import Iterable, Iterator
import regex as re


class BPEtokenizer:
    def __init__(self, vocab: dict[int, bytes], 
                merges: list[tuple[bytes, bytes]],
                special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.cache = {}
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.pair_to_priority = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, 
                   vocab_filepath: str, 
                   merges_filepath: str, 
                   special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab = {}
            for line in f:
                idx, token = line.strip().split(b"\t", 1)
                vocab[int(idx)] = token
                
        with open(merges_filepath, "rb") as f:
            merges = []
            for line in f:
                token1, token2 = line.strip().split(b" ")
                merges.append((token1, token2))

        return cls(vocab, merges, special_tokens)
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            id_list = self.encode(line)

            yield from id_list
    
    def encode(self, text: str) -> list[int]:
        final_ids = []

        # Pretokenize the text
        pre_tokens = self.pretokenize(text, self.special_tokens if self.special_tokens else [])

        encoded_special_tokens = [token.encode("utf-8") for token in self.special_tokens] if self.special_tokens else []
        
        # Apply BPE to get tokens
        for token in pre_tokens:
            if token in encoded_special_tokens:
                final_ids.append(self.token_to_id[token])
            elif token in self.cache.keys():
                final_ids.extend(self.cache[token])
            else:
                parts = self.bpe_merge(token)
                ids = []
                for part in parts:
                    ids.append(self.token_to_id[part])
                final_ids.extend(ids)
                self.cache[token] = ids

        return final_ids

    def pretokenize(self, chunk: str, special_tokens:list[str]) -> list[bytes]:
        
        # Sort the special tokens by length
        if special_tokens:
            special_tokens.sort(key=len, reverse=True)

        # Split the corpus on special tokens
        escaped_tokens = []
        for token in special_tokens:
            escaped_tokens.append(re.escape(token))

        split_on_token = '|'.join(escaped_tokens)
        if split_on_token:
            split_on_token_with_capture = f'({split_on_token})'
            splited_chunk = re.split(split_on_token_with_capture,chunk)
        else:
            splited_chunk = [chunk]

        # Pre-tokenize by PAT
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokens = []

        for token in splited_chunk:
            if not token:
                continue
            elif special_tokens and token in set(special_tokens):
                pre_tokens.append(token.encode("utf-8"))
            else:
                for match in re.finditer(PAT, token):
                    pre_tokens.append(match.group().encode("utf-8"))

        return pre_tokens
    
    def bpe_merge(self, token: bytes) -> list:

        # Turn token into iterable list
        parts = [bytes([b]) for b in token]

        # Loop to merge all mergable pairs
        while True:
            best_priority = len(self.merges)
            best_pair = None

            # Look for mergable pair
            for i in range(len(parts)-1):
                current_pair = (parts[i], parts[i+1])

                # If mergable, compare the priority to get the best one
                if current_pair in self.pair_to_priority.keys():
                    pair_priority = self.pair_to_priority[current_pair]
                    if pair_priority < best_priority:
                        best_priority = pair_priority
                        best_pair = current_pair
                else:
                    pass
            
            # If no pair to merge, break
            if best_pair == None:
                break
            
            # Update the token list
            i = 0
            while i < len(parts):
                if i+1 < len(parts) and (parts[i], parts[i+1]) == best_pair:
                    parts[i] = parts[i] + parts[i+1]
                    del parts[i+1]
                else:
                    i += 1


        return parts

    def decode(self, ids: list[int]) -> str:
        text = b"".join([self.vocab[id] for id in ids]).decode("utf-8", errors="replace")
        return text