from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from os import PathLike

import regex


# 作业测试数据使用的 GPT-2 预分词正则：先按词、数字、标点和空白分段，
# 后续 BPE 只会在同一个预分词片段内部合并字节。
_PRETOKEN_PATTERN = regex.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)
Pair = tuple[bytes, bytes]


def _split_special_tokens(text: str, special_tokens: list[str]) -> Iterator[tuple[str, bool]]:
    """逐段产出文本，并标记该段是否为特殊 token。"""
    if not special_tokens:
        yield text, False
        return

    # 长 token 优先，确保重叠的特殊 token 有确定的匹配结果。
    pattern = regex.compile(
        "(" + "|".join(regex.escape(token) for token in sorted(special_tokens, key=len, reverse=True)) + ")"
    )
    for fragment in pattern.split(text):
        if fragment:
            yield fragment, fragment in special_tokens


def _merge_pair(symbols: list[bytes], pair: Pair) -> list[bytes]:
    """将 token 序列中每个不重叠的 ``pair`` 合并成一个新 token。"""
    merged: list[bytes] = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
            merged.append(pair[0] + pair[1])
            index += 2
        else:
            merged.append(symbols[index])
            index += 1
    return merged


class Tokenizer:
    """与提供的 GPT-2 测试词表兼容的字节级 BPE 分词器。"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Pair],
        special_tokens: list[str] | None = None,
    ) -> None:
        # vocab 的原始方向是 id -> bytes；编码时频繁反查，因此额外建立
        # bytes -> id 的反向索引。merges 在列表中的下标就是合并优先级。
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])
        self._token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._special_token_ids = {
            token: self._token_to_id[token.encode("utf-8")]
            for token in self.special_tokens
            if token.encode("utf-8") in self._token_to_id
        }

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        # BPE 的最小单位不是字符，而是 UTF-8 的每一个字节；中文、emoji
        # 都会在这里自然展开为多个字节。
        symbols = [bytes([byte]) for byte in pretoken.encode("utf-8")]

        while len(symbols) > 1:
            # 当前序列中可能有多个可合并 pair。选择训练时最早创建的那个，
            # 即 rank 最小者；这正是 merges 列表在编码阶段的作用。
            ranked_pairs = [
                (self._merge_ranks[pair], pair)
                for pair in zip(symbols, symbols[1:])
                if pair in self._merge_ranks
            ]
            if not ranked_pairs:
                break
            _, pair = min(ranked_pairs, key=lambda item: item[0])
            # 同一个 pair 在一个片段里可能出现多次，必须在这一步全部合并。
            symbols = _merge_pair(symbols, pair)

        # 片段已无法继续合并，再把最终 bytes token 映射成模型使用的整数 id。
        return [self._token_to_id[symbol] for symbol in symbols]

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        # 特殊 token 先切出来，避免它内部的字符被预分词和 BPE 拆开。
        for fragment, is_special in _split_special_tokens(text, self.special_tokens):
            if is_special:
                token_ids.append(self._special_token_ids[fragment])
            else:
                # 每个预分词片段彼此独立，不能跨片段应用 merge。
                for pretoken in _PRETOKEN_PATTERN.findall(fragment):
                    token_ids.extend(self._encode_pretoken(pretoken))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """逐段编码，不会先把整个输入迭代器读入内存。"""
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: Iterable[int]) -> str:
        # token 可以分别代表单字节或多字节；先拼回原始字节流，再一次性 UTF-8 解码。
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")


def train_bpe(
    input_path: str | PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[Pair]]:
    """以 GPT-2 风格预分词训练字节级 BPE 词表。"""
    unique_special_tokens = list(dict.fromkeys(special_tokens))
    if vocab_size < 256 + len(unique_special_tokens):
        raise ValueError("vocab_size must fit the 256 byte tokens and all special tokens")

    # 训练只读取一次完整语料；之后使用“唯一预分词片段 -> 出现次数”的压缩表示。
    with open(input_path, encoding="utf-8") as corpus:
        text = corpus.read()

    pretoken_counts: Counter[bytes] = Counter()
    for fragment, is_special in _split_special_tokens(text, unique_special_tokens):
        if not is_special:
            # 特殊 token 不参与统计，因此 BPE 不会产生包含它们一部分的普通 token。
            pretoken_counts.update(token.encode("utf-8") for token in _PRETOKEN_PATTERN.findall(fragment))

    # 训练从 256 个可能的单字节 token 开始；特殊 token 直接占用独立词表项，
    # 不需要、也不能通过 BPE merge 学出来。
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    for token in unique_special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # 每个原始预分词片段各占一个位置；记录其出现频次。这样每次合并只需
    # 更新受该 pair 影响的词，同时仍能按原始语料频次统计。
    words = [[bytes([byte]) for byte in pretoken] for pretoken in pretoken_counts]
    frequencies = [pretoken_counts[pretoken] for pretoken in pretoken_counts]
    pair_counts: Counter[Pair] = Counter()
    pair_to_words: defaultdict[Pair, set[int]] = defaultdict(set)

    # 建立训练循环的两张核心表：
    # pair_counts[pair] 是 pair 的全语料加权频次；
    # pair_to_words[pair] 是它出现在哪些 words 中，用于局部更新。
    for word_id, symbols in enumerate(words):
        for pair, occurrences in Counter(zip(symbols, symbols[1:])).items():
            pair_counts[pair] += frequencies[word_id] * occurrences
            pair_to_words[pair].add(word_id)

    merges: list[Pair] = []
    while len(vocab) < vocab_size and pair_counts:
        # 频次相同时按作业要求选字典序最大的字节对。
        pair = max(pair_counts, key=lambda candidate: (pair_counts[candidate], candidate))
        # 此时 pair 正式成为新 token；merges 追加顺序也就是编码优先级。
        merges.append(pair)
        vocab[len(vocab)] = pair[0] + pair[1]

        # 只处理包含这个 pair 的词，不重新扫描整个语料库。
        affected_word_ids = list(pair_to_words[pair])
        for word_id in affected_word_ids:
            old_symbols = words[word_id]
            # 先撤销该词合并前对全局统计表的贡献。
            old_pairs = Counter(zip(old_symbols, old_symbols[1:]))
            new_symbols = _merge_pair(old_symbols, pair)
            # 再加入合并后新产生的相邻 pair；例如 a+b+c 合并 a+b 后，会形成 ab+c。
            new_pairs = Counter(zip(new_symbols, new_symbols[1:]))
            frequency = frequencies[word_id]

            for old_pair, occurrences in old_pairs.items():
                pair_counts[old_pair] -= frequency * occurrences
                if pair_counts[old_pair] == 0:
                    del pair_counts[old_pair]
                pair_to_words[old_pair].discard(word_id)

            words[word_id] = new_symbols
            # 将新词的 pair 贡献写回两张核心表，供下一轮选择最佳 merge。
            for new_pair, occurrences in new_pairs.items():
                pair_counts[new_pair] += frequency * occurrences
                pair_to_words[new_pair].add(word_id)

    return vocab, merges
