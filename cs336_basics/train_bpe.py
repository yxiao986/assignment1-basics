from collections import Counter
from .pretokenization_example import find_chunk_boundaries
from itertools import repeat
import multiprocessing
import regex as re

def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]
) -> tuple[dict[int,bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Args:
        input_path(str): path of training data
        vocab_size(int): final size of vocabulary
        special_tokens(list[str]): special tokens that we need to add to vocabulary
    
        Returns:
            tuple[dict[int, bytes], list[tuple[bytes, bytes]]]
            - vocab: mapping dict from token ID (int) to token(bytes)
            - merges: BPE merging rules in order
    """
    # initialize vocabulary
    vocab = init_vocab(special_tokens)

    # pretokenization and word frequency counting
    pre_token_counts = parallel_pre_tokenize(input_path, special_tokens)

    # iteratively calculate and merge
    vocab, merges = compute_merges(vocab, pre_token_counts, vocab_size)

    # return vocab and merges
    return vocab, merges

def init_vocab(
        special_tokens:list[str],
) -> dict[int,bytes]:
    vocab = {i: bytes([i]) for i in range(256)}
    special_token_id = len(vocab)
    for token in special_tokens:
        vocab[special_token_id] = token.encode("utf-8")
        special_token_id += 1
    return vocab

def parallel_pre_tokenize(
        input_path:str,
        special_tokens:list[str]
) -> dict[tuple[bytes], int]:
    
    with open(input_path, "rb") as f:
    
        # parallel with multiprocessing
        num_process =  4 # multiprocessing.cpu_count() 
        boundaries = find_chunk_boundaries(f, num_process, special_tokens[0].encode("utf-8"))

        with multiprocessing.Pool() as pool:
            args = zip(repeat(input_path), boundaries[:-1], boundaries[1:], repeat(special_tokens))
            pre_token_counts = pool.map(process_chunk, args)
            #pre_token_counts = [process_chunk(arg) for arg in args]

        # merge the result from different process
        total_pre_token_counts = Counter()
        for counts in pre_token_counts:
            total_pre_token_counts += counts

        return total_pre_token_counts



def process_chunk(
        args:tuple
) -> Counter:
    input_path,start, end, special_tokens = args

    with open(input_path,'rb') as f:

        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")

        # split the corpus on special tokens
        escaped_tokens = []
        for token in special_tokens:
            escaped_tokens.append(re.escape(token))

        split_on_token = '|'.join(escaped_tokens)
        stripped_chunk = re.split(split_on_token,chunk)

        # pre-tokenize by PAT
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokens = []
        for token in stripped_chunk:
            pre_token = re.finditer(PAT, token)
            pre_tokens += pre_token

        # encode the tokens
        encoded_pre_tokens = []
        for token in pre_tokens:
            encoded_pre_tokens.append(token.group().encode("utf-8"))

        # count the pre_tokens
        tuple_pre_tokens = (tuple(bytes([b]) for b in token) for token in encoded_pre_tokens)
        pre_token_counts = Counter(tuple_pre_tokens)

        return pre_token_counts

        

def compute_merges(
        vocab:dict[int, bytes],
        pre_token_counts:dict[tuple[bytes], int],
        vocab_size:int
)   -> tuple[dict[int, bytes],list[tuple[bytes,bytes]]] :
        
        merges = []
        pair_counts = Counter()

        # iterate all bytes pairs, count their freq
        for token, freq in pre_token_counts.items():
            for i in range(len(token)-1):
                pair_counts[(token[i],token[i+1])] += freq

        while len(vocab) < vocab_size and pair_counts:

            # find the most frequent pair
            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            merged_pair = best_pair[0] + best_pair[1]

            # update vocab and merge
            merges.append(best_pair)
            vocab[len(vocab)] = merged_pair

            # update pair freq count
            new_pre_token_counts = Counter()
            for token, freq in pre_token_counts.items():
                new_token = ()
                i = 0
                while i < len(token):
                    if i+1<len(token) and token[i] == best_pair[0] and token[i+1] == best_pair[1]:

                        # merge pair
                        new_token += (merged_pair,)
                        
                        # update left neighbour
                        if i-1 >= 0:
                            pair_counts[(token[i-1],token[i])] -= freq
                            pair_counts[(token[i-1],merged_pair)] += freq

                        # update right pair
                        if i+2 < len(token):
                            pair_counts[(token[i+1],token[i+2])] -= freq
                            pair_counts[(merged_pair,token[i+2])] += freq

                        i += 2

                    else:
                        new_token += (token[i],)
                        i += 1
                                        
                new_pre_token_counts[new_token] += freq
            
            pre_token_counts = new_pre_token_counts
            del pair_counts[best_pair]

        return vocab, merges



            

if __name__ == '__main__':
    # test init_vocab
    special_tokens = ["<|endoftext|>"]
    vocab = init_vocab(special_tokens)

    # test parallel_pre_tokenize
    pre_tokens = parallel_pre_tokenize("data/owt_valid.txt", special_tokens)
    print(pre_tokens)

