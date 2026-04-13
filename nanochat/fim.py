import random as _random

from nanochat.tokenizer import SPECIAL_TOKENS

def _get_fim_token_ids(tokenizer):
    vocab_size = tokenizer.get_vocab_size()
    n = len(SPECIAL_TOKENS)
    def _id(index):
        return vocab_size - n + index
    return {
        "bos":    _id(0),
        "prefix": _id(9),
        "suffix": _id(10),
        "middle": _id(11),
    }


def apply_fim(token_ids: list[int], tokenizer, fim_rate=0.5, spm_rate=0.5, rng=None) -> list[int]:
    """
    Probabilistically transform a tokenized document into FIM format.

    Args:
        token_ids: Tokenized document starting with BOS token
        tokenizer: The RustBPETokenizer instance (has encode/decode/get_vocab_size)
        fim_rate: Probability of applying FIM (0.5 = 50% of documents get FIM)
        spm_rate: Among FIM docs, probability of SPM format vs PSM (0.5 = equal mix)
        rng: random.Random instance for reproducibility

    Returns:
        Transformed token_ids (FIM format) or original token_ids (unchanged)

    FIM formats:
        PSM: [BOS, FIM_PREFIX] + prefix + [FIM_SUFFIX] + suffix + [FIM_MIDDLE] + middle
        SPM: [BOS, FIM_SUFFIX] + suffix + [FIM_PREFIX] + prefix + [FIM_MIDDLE] + middle
    """
    if rng is None:
        rng = _random.Random()

    if rng.random() > fim_rate:
        return token_ids

    if len(token_ids) < 50:
        return token_ids

    tids = _get_fim_token_ids(tokenizer)
    bos = tids["bos"]
    fim_prefix = tids["prefix"]
    fim_suffix = tids["suffix"]
    fim_middle = tids["middle"]

    text = tokenizer.decode(token_ids[1:])
    n = len(text)
    lo = int(0.1 * n)
    hi = int(0.9 * n)
    p1 = rng.randint(lo, hi)
    p2 = rng.randint(lo, hi)
    if p1 > p2:
        p1, p2 = p2, p1

    prefix_text = text[:p1]
    middle_text = text[p1:p2]
    suffix_text = text[p2:]

    prefix_ids = tokenizer.encode(prefix_text)
    middle_ids = tokenizer.encode(middle_text)
    suffix_ids = tokenizer.encode(suffix_text)

    if rng.random() < spm_rate:
        # SPM: [BOS, FIM_SUFFIX] + suffix + [FIM_PREFIX] + prefix + [FIM_MIDDLE] + middle
        result = [bos, fim_suffix] + suffix_ids + [fim_prefix] + prefix_ids + [fim_middle] + middle_ids
    else:
        # PSM: [BOS, FIM_PREFIX] + prefix + [FIM_SUFFIX] + suffix + [FIM_MIDDLE] + middle
        result = [bos, fim_prefix] + prefix_ids + [fim_suffix] + suffix_ids + [fim_middle] + middle_ids

    return result


def make_fim_transform(tokenizer, fim_rate=0.5, spm_rate=0.5, seed=42):
    """
    Factory function that returns a callable suitable for the dataloader's fim_transform_fn parameter.
    The returned function takes token_ids (list[int]) and returns transformed token_ids.
    Uses a per-call Random instance seeded from the base seed for reproducibility.
    """
    base_rng = _random.Random(seed)

    def transform(token_ids: list[int]) -> list[int]:
        child_seed = base_rng.randint(0, 2**31 - 1)
        rng = _random.Random(child_seed)
        return apply_fim(token_ids, tokenizer, fim_rate=fim_rate, spm_rate=spm_rate, rng=rng)

    return transform
