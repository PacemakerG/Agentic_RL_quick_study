# CS336 作业 1（基础）：从零构建 Transformer 语言模型

> 版本：26.0.3
>
> Stanford CS336 课程组，2026 年春季
>
> 本文是 `cs336_assignment1_basics.pdf` 的完整中文翻译，仅翻译原作业说明，不包含题目答案。原文已有的示例代码、函数接口、命令和正则表达式保持不变。

## 1 作业概览

本作业要求你从零构建训练一个标准 Transformer 语言模型所需的全部组件，并实际训练一些模型。

### 需要实现的内容

1. 字节对编码（BPE）分词器（第 2 节）。
2. Transformer 语言模型（第 3 节）。
3. 交叉熵损失函数和 AdamW 优化器（第 4 节）。
4. 训练循环，并支持模型与优化器状态的序列化和加载（第 5 节）。

### 需要运行的内容

1. 在 TinyStories 数据集上训练 BPE 分词器。
2. 使用训练好的分词器编码数据集，将其转换为整数 ID 序列。
3. 在 TinyStories 数据集上训练 Transformer 语言模型。
4. 使用训练好的 Transformer LM 生成样本，并用困惑度进行评估。
5. 在 OpenWebText 上训练模型，并把达到的困惑度提交到排行榜。

### 可以使用的内容

课程希望你从零构建每个组件。除下列内容外，不得使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的任何定义：

- `torch.nn.Parameter`；
- `torch.nn` 中的容器类，例如 `Module`、`ModuleList`、`Sequential` 等；
- `torch.optim.Optimizer` 基类。

可以使用 PyTorch 中的其他定义。如果不确定某个函数或类是否允许使用，可以在课程 Slack 中询问。拿不准时，应思考它是否破坏了本作业“从零实现”的精神。

### 关于 AI 工具的声明

AI 可以全自动解决作业中的许多部分，这会使学生更难深入参与并真正学会课程内容。

课程允许使用 AI 回答高层概念问题，或查询函数签名、库 API 等底层编程文档；但不允许使用 AI 实现作业的任何部分，包括 Cursor Agents、Codex、Claude Code 等编码代理和 Cursor Tab、GitHub Copilot 等 AI 自动补全。使用 AI 代理时，应确保代理遵守仓库中的 `AGENTS.md`；使用聊天机器人时，也应把该文件中的提示一并提供。

课程强烈建议完成作业时关闭 AI 自动补全。非 AI 的函数名自动补全可以正常使用。往届学生反馈，关闭 AI 自动补全有助于更深入地学习材料。完整规定参见课程的 AI Policy。

### 代码结构

作业代码和说明位于：<https://github.com/stanford-cs336/assignment1-basics>。请克隆该仓库；如有更新，课程组会通知你拉取最新版。

1. `cs336_basics/*`：编写自己代码的位置。这里最初没有代码，你可以从零组织实现。
2. `adapters.py`：测试所需的功能接口。对于 scaled dot-product attention 等每项功能，在对应 adapter 中调用自己的代码。adapter 只应是胶水代码，不应包含实质逻辑。
3. `test_*.py`：必须通过的测试。测试会调用 `adapters.py` 中定义的接口。不要修改测试文件。

### 提交方式

运行 `make_submission.sh` 生成提交压缩包。如果存在不希望打包的大型数据或 checkpoint，请把它们加入脚本的排除列表。

向 Gradescope 提交：

- `writeup.pdf`：排版后的所有书面题答案；
- `code.zip`：编写的全部代码。

排行榜提交地址：<https://github.com/stanford-cs336/assignment1-basics-leaderboard>。详细规则见该仓库 README。

### 数据集

作业使用两个预处理数据集：TinyStories [1] 和 OpenWebText [2]，两者都是单个大型纯文本文件。

正式选课学生按课程 compute guide 下载；自行学习者按作业仓库 README 中的命令下载。

```bash
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### 低资源提示

作业说明中会用蓝色框提供低资源建议，例如缩小数据集、缩小模型，或说明如何在 Mac 集成 GPU 和 CPU 上训练。即使拥有课程计算资源，这些建议也能帮助快速迭代和节省时间。

以课程组实现为例，在配备 36 GB 内存的 Apple M4 Max 上，用 MPS 训练一个能生成较流畅文本的 LM 不到 5 分钟，用 CPU 约 30 分钟。只要笔记本较新且实现正确高效，就能训练一个可以生成简单儿童故事的小型 LM。后续章节会说明 CPU 或 MPS 下需要进行的调整。

## 2 字节对编码（BPE）分词器

本部分训练并实现一个字节级 BPE 分词器 [3, 4]。任意 Unicode 字符串会先表示为字节序列，再在该序列上训练 BPE。之后使用分词器把文本编码为整数 token 序列，用于语言建模。

### 2.1 Unicode 标准

Unicode 把字符映射到整数码点。截至 2025 年 9 月发布的 Unicode 17.0，共定义 172 种书写系统中的 159,801 个字符。例如，“s”的码点是 115，通常记为 U+0073；“牛”的码点是 29275。Python 的 `ord()` 可把单个字符转换为整数，`chr()` 可把整数码点转换为对应字符。

```python
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

**题目（unicode1）：理解 Unicode（1 分）**

1. `chr(0)` 返回什么 Unicode 字符？交付物：一句话。
2. 该字符的字符串表示 `__repr__()` 与打印表示有什么不同？交付物：一句话。
3. 该字符出现在文本中会发生什么？可在 Python 解释器中尝试以下内容，观察结果是否符合预期。交付物：一句话。

```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

### 2.2 Unicode 编码

直接在 Unicode 码点上训练分词器并不现实，因为词表会非常大（约 15 万项）且稀疏。Unicode 编码把字符转换成字节序列。Unicode 定义 UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网主流编码，超过 98% 的网页使用它。

Python 可用 `encode()` 将字符串编码为 UTF-8；迭代 `bytes` 对象或调用 `list()` 可以查看 0 到 255 的字节值；`decode()` 可将 UTF-8 字节串解码回字符串。一个字节不一定对应一个 Unicode 字符，例如包含日文字符的 13 字符字符串可能编码成 23 个 UTF-8 字节。

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171,
 227, 129, 161, 227, 129, 175, 33]
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

将 Unicode 码点转换为字节序列，本质上是把拥有 159,801 个有效值的 21 位整数序列转换为 0 到 255 的整数序列。256 项的字节词表更易管理，而且任何输入都能表示为字节，因此没有词表外 token。

**题目（unicode2）：Unicode 编码（3 分）**

1. 为什么训练分词器时更倾向于 UTF-8，而不是 UTF-16 或 UTF-32？可比较不同字符串的编码结果。交付物：一到两句话。
2. 考虑下面这个意图把 UTF-8 字节串解码为 Unicode 字符串、但实际错误的函数。为什么它不正确？给出会产生错误结果的输入字节串。交付物：示例加一句解释。

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```
3. 给出一个不能解码为任何 Unicode 字符的两字节序列。交付物：示例加一句解释。

### 2.3 子词分词

字节级分词消除了词表外问题，但会产生很长的输入序列。一个十词句子在词级模型中可能只有十个 token，在字符或字节级模型中却可能有五十个以上。这会增加每一步计算量，也会制造更长的依赖。

子词分词位于词级和字节级之间：它以更大的词表换取更强的压缩。例如，若 `b'the'` 在训练语料中经常出现，就可以将其作为一个词表项，把三个字节 token 压缩为一个 token。

BPE 是一种压缩算法：反复将最高频的相邻字节对合并为一个新的 token。它通过加入高频子词来最大化压缩；足够高频的完整单词也可能成为单个 token。构建 BPE 词表的过程称为训练分词器。

### 2.4 BPE 分词器训练

BPE 训练包含三个主要步骤。

#### 词表初始化

词表是字节串 token 到整数 ID 的一一映射。字节级 BPE 的初始词表包括全部 256 个字节值，因此初始大小为 256。

#### 预分词

如果每次合并前都完整扫描语料并重新统计相邻字节，成本很高。直接在整份语料中合并还可能得到仅标点不同、却拥有完全不同 ID 的 token，例如 `dog!` 与 `dog.`。

为此先进行粗粒度预分词。若预 token `text` 出现十次，在统计 `t` 和 `e` 的邻接次数时可以直接增加十，而不必逐处扫描。每个预 token 表示为 UTF-8 字节序列。

原始 BPE 只按空格切分；SentencePiece 系列分词器中仍可见这种做法。现代分词器通常采用源自 GPT-2 的正则预分词。本作业使用以下模式：

```text
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

例如：

```python
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

实际代码应使用 `re.finditer`，避免在统计时保存所有预 token。

#### 计算 BPE 合并

把语料变为预 token 和 UTF-8 字节序列后，反复统计相邻 token 对，选择最高频对 `(A, B)`，将其每次出现替换为新 token `AB`，并将 `AB` 加入词表。最终词表大小等于初始 256 项加合并次数和特殊 token 数。不得跨预 token 边界统计或合并。

若多个 token 对频率相同，以字典序较大的 pair 打破平局。例如 `("A", "B")`、`("A", "C")`、`("B", "ZZ")`、`("BA", "A")` 同频时，选择 `("BA", "A")`。

#### 特殊 token

`<|endoftext|>` 等字符串用于编码文档边界等元数据。它们应始终保留为单个 token，并拥有固定 ID。字节级 BPE 不加入原始 BPE 中的词尾 token，因为空白和标点本身已经在字节词表中，学习出的合并会自然反映词边界。

#### 示例（bpe_example）：BPE 训练

考虑下面的语料，词表另含特殊 token `<|endoftext|>`：

```text
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

初始词表是该特殊 token 和 256 个字节。

为简化演示，按空格预分词得到频率表 `{low: 5, lower: 2, widest: 3, newest: 6}`。用 `dict[tuple[bytes, ...], int]` 表示时，可写成 `{(l,o,w): 5, …}`。注意，Python 中即使单个字节也是 `bytes` 对象；正如它没有单字符 `char` 类型，也没有表示单个字节的独立 `byte` 类型。

第一轮统计相邻 pair 得到：

```text
{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3,
 es: 9, st: 9, ne: 6, ew: 6}
```

`('e','s')` 与 `('s','t')` 同为最高频，按字典序选择 `('s','t')`，把预 token 变为：

```text
{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}
```

第二轮最高频的是 `(e, st)`，计数为 9，合并后得到：

```text
{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}
```

完整合并序列为：

```text
['s t', 'e st', 'o w', 'l ow', 'w est', 'n e',
 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']
```

若只执行前六次合并，词表除特殊 token 和 256 个字节外，还包括 `st`、`est`、`ow`、`low`、`west`、`ne`；此时 `newest` 被编码为 `[ne, west]`。

### 2.5 BPE 训练实验

先查看 TinyStories 数据，了解其内容，再训练字节级 BPE。

预分词通常是主要瓶颈，可以用 `multiprocessing` 并行化。并行切分语料时，应让 chunk 边界位于特殊 token 开头。作业允许原样使用仓库中的 `pretokenization_example.py` 来获取边界。由于不允许跨文档合并，这种切分对本作业始终有效。

运行正则预分词前，应先在特殊 token 处切分语料，并从各 chunk 中移除特殊 token。特殊 token 是硬边界，但不参与合并统计。实现时需正确转义特殊 token 中可能出现的 `|`；`test_train_bpe_special_tokens` 会检查这一点。

朴素合并每轮都遍历全部 pair，速度很慢。一次合并后，只有与该 pair 重叠的 pair 计数会变化，因此可索引所有 pair，并增量更新计数。合并步骤在 Python 中本身不能并行。使用 `cProfile` 或 `py-spy` 查找瓶颈。

低资源调试时，先在 TinyStories 验证集等较小数据上训练。验证集约 2.2 万篇文档，训练集约 212 万篇。调试规模要足以复现完整任务瓶颈，但不能大到拖慢迭代。

**题目（train_bpe）：BPE 分词器训练（15 分）**

实现一个函数，根据文本文件路径训练字节级 BPE。至少支持：

- `input_path: str`：训练文本路径；
- `vocab_size: int`：最终最大词表大小，包含初始字节、合并项和特殊 token；
- `special_tokens: list[str]`：加入词表的特殊字符串。它们是禁止跨越的硬边界，但不计入合并统计。

返回：

- `vocab: dict[int, bytes]`：token ID 到 token 字节串的映射；
- `merges: list[tuple[bytes, bytes]]`：按创建顺序排列的合并 pair。

完成 `adapters.run_train_bpe` 后运行 `uv run pytest tests/test_train_bpe.py`，实现应通过全部测试。可选扩展是用 C++ 或 Rust 实现关键部分，但需处理 Python 内存交互和构建说明。GPT-2 正则在许多正则引擎中支持不佳；课程组验证 Oniguruma 可用，而 Python `regex` 甚至更快。

**题目（train_bpe_tinystories）：在 TinyStories 上训练 BPE（2 分）**

1. 使用最大词表 10,000 训练字节级 BPE，把 `<|endoftext|>` 加入词表，将词表和 merges 序列化。报告训练时间、内存、最长 token，并判断是否合理。资源：不超过 30 分钟、无需 GPU、不超过 30 GB 内存。交付物：一到两句话。
2. 对代码进行 profiling，指出训练过程最耗时的部分。交付物：一到两句话。

提示：利用多进程预分词、文档分隔 token 和特殊 token 的预处理，训练应可控制在两分钟内。

**题目（train_bpe_expts_owt）：在 OpenWebText 上训练 BPE（2 分）**

1. 使用最大词表 32,000 训练字节级 BPE，序列化词表和 merges，报告最长 token 并判断是否合理。资源：不超过 12 小时、无需 GPU、不超过 100 GB 内存。交付物：一到两句话。
2. 比较 TinyStories 与 OpenWebText 分词器。交付物：一到两句话。

### 2.6 BPE 编码与解码

训练得到词表和合并规则后，实现一个可从文件加载它们、并在文本与 token ID 之间转换的分词器。

#### 2.6.1 编码

编码分两步：先按训练时相同的方法预分词，把每个预 token 表示为 UTF-8 字节；再按照 merges 的创建顺序，在每个预 token 内应用合并，不跨边界。

示例输入是 `the cat ate`，词表为：

```text
{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't',
 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
```

学习出的 merges 为：

```text
[(b't', b'h'), (b' ', b'c'), (b' ', b'a'),
 (b'th', b'e'), (b' a', b't')]
```

输入先被预分为 `['the', ' cat', ' ate']`。`the` 从 `[b't', b'h', b'e']` 依次合并为 `[b'th', b'e']`、`[b'the']`，对应 `[9]`；` cat` 变为 `[b' c', b'a', b't']`，对应 `[7,1,5]`；` ate` 变为 `[b' at', b'e']`，对应 `[10,3]`。最终编码为 `[9,7,1,5,10,3]`。编码还必须正确处理构造分词器时提供的特殊 token。

对于不能整体装入内存的大文件，应分块流式处理，使内存复杂度保持为常数。必须确保 token 不跨 chunk 边界，否则结果会与整段编码不同。

#### 2.6.2 解码

解码时，按 ID 查询对应字节串，拼接后解码为 Unicode。任意 ID 序列不一定构成合法 Unicode；遇到非法字节时，应使用官方替换字符 U+FFFD。`bytes.decode(errors='replace')` 可完成此处理。

**题目（tokenizer）：实现分词器（15 分）**

实现 `Tokenizer` 类，根据词表和 merges 编码、解码，并支持用户提供的特殊 token；若特殊 token 尚不在词表中，则追加进去。建议提供：

- 构造函数：从词表、merges 和可选特殊 token 创建分词器；
- `from_files`：从序列化文件创建分词器；
- `encode`：文本转 ID 列表；
- `encode_iterable`：从字符串 iterable 惰性地产生 ID，以支持大文件；
- `decode`：ID 列表转文本。

完成 `adapters.get_tokenizer` 后运行 `uv run pytest tests/test_tokenizer.py`，实现应通过全部测试。

### 2.7 分词器实验

**题目（tokenizer_experiments）：分词器实验（4 分）**

1. 从 TinyStories 和 OpenWebText 各采样十篇文档，分别用 10K 和 32K 词表分词器编码，计算字节数/token 数压缩比。交付物：一到两句话。
2. 用 TinyStories 分词器编码 OpenWebText 样本，比较压缩比或定性描述结果。交付物：一到两句话。
3. 估算分词器吞吐量（字节/秒），并估算编码 825 GB Pile 数据集所需时间。交付物：一到两句话。
4. 将两套数据的训练集和开发集编码为 ID 序列，建议保存为 `uint16` NumPy 数组。解释 `uint16` 为什么合适。交付物：一到两句话。

## 3 Transformer 语言模型架构

语言模型输入形状为 `(batch_size, sequence_length)` 的整数 token ID，输出形状为 `(batch_size, sequence_length, vocab_size)` 的词表概率分布，每个位置预测下一个 token。训练时用这些预测计算交叉熵；生成时取最后一个位置的分布，选择或采样下一个 token，追加到输入后重复。

本部分从零构建 Transformer LM，先介绍整体结构，再介绍各组件。

### 3.1 Transformer LM

token ID 先经输入 embedding 变成稠密向量，再通过 `num_layers` 个 Transformer Block，最后通过可学习线性投影（输出 embedding 或 LM head）得到下一个 token 的 logits。

图 1 的结构为：输入 → Token Embedding → 多个 Transformer Block → Norm → Linear/LM Head → Softmax → 输出概率。

Token Embedding 将 `(batch_size, sequence_length)` 的整数 Tensor 转换为 `(batch_size, sequence_length, d_model)` 的向量序列。

标准 decoder-only Transformer 包含多个相同 Block。每个 Block 的输入输出均为 `(batch_size, sequence_length, d_model)`；self-attention 聚合跨位置的信息，前馈层逐位置进行非线性变换。最后一个 Block 后还要归一化，再由线性变换产生 logits。

图 2 的 pre-norm Block 包含两条子层：输入先经 Norm、带 RoPE 的因果多头自注意力并加残差；随后再经 Norm、逐位置前馈网络并加残差。

### 3.2 Batching、Einsum 与高效计算

Transformer 会在样本、序列位置、attention head 等 batch 类维度上重复同一种计算。许多 PyTorch 操作允许 Tensor 开头存在额外 batch 维度，并高效广播。

例如，形状 `(batch_size, sequence_length, d_model)` 的数据与 `(d_model, d_model)` 的矩阵相乘时，前两个维度自然作为 batch。函数应容忍任意额外 batch 维度，并把它们放在形状前部。

只用 `view`、`reshape`、`transpose` 会让代码难读。`einsum` 能执行任意维度的张量收缩，`rearrange` 能重排、拼接和拆分维度。课程建议初学者使用 `einops`，熟悉后学习 `einx`。PDF 用批量矩阵乘法、批量图像明暗变换和像素混合三个例子展示其自描述性。`einx` 不如 `einops` 成熟，遇到问题可以退回普通 PyTorch 配合 `einops`。

**示例（einstein_example1）：使用 `einops.einsum` 的批量矩阵乘法**

```python
import torch
from einops import rearrange, einsum

# 基本实现
Y = D @ A.T

# Einsum 写法会直接记录维度语义
Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")

# D 可以拥有任意前导维度，而 A 的形状受到约束
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
```

**示例（einstein_example2）：使用 `einops.rearrange` 广播**

有一批图像，需要按十个缩放系数为每幅图产生十个变暗版本：

```python
images = torch.randn(64, 128, 128, 3)  # (batch, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)

dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
images_rearr = rearrange(
    images, "b height width channel -> b 1 height width channel"
)
dimmed_images = images_rearr * dim_value

# 或一次完成
dimmed_images = einsum(
    images,
    dim_by,
    "batch height width channel, dim_value -> batch dim_value height width channel",
)
```

**示例（einstein_example3）：使用 `einops.rearrange` 混合像素**

一批图像形状为 `(batch, height, width, channel)`，要在每个通道内独立地对所有像素执行线性变换。矩阵 $B$ 的形状是 `(height * width, height * width)`。

```python
channels_last = torch.randn(64, 32, 32, 3)
B = torch.randn(32 * 32, 32 * 32)

# 普通 view + transpose
channels_last_flat = channels_last.view(
    -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
)
channels_first_flat = channels_last_flat.transpose(1, 2)
channels_first_flat_transformed = channels_first_flat @ B.T
channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)

# 使用 einops
height = width = 32
channels_first = rearrange(
    channels_last,
    "batch height width channel -> batch channel (height width)",
)
channels_first_transformed = einsum(
    channels_first,
    B,
    "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out",
)
channels_last_transformed = rearrange(
    channels_first_transformed,
    "batch channel (height width) -> batch height width channel",
    height=height,
    width=width,
)

# 使用 einx.dot 一次完成
channels_last_transformed = einx.dot(
    "batch row_in col_in channel, (row_out col_out) (row_in col_in)"
    "-> batch row_out col_out channel",
    channels_last,
    B,
    col_in=width,
    col_out=width,
)
```

普通实现可以在前后增加形状注释，但仍然笨重且容易出错；einsum 表达式本身就起到了文档作用。对于其他 Tensor，还可以使用 `jaxtyping` 等 Tensor 类型标注。

#### 3.2.1 数学记号与内存顺序

机器学习论文常用行向量：

$$y=xW^\top,\qquad W\in\mathbb{R}^{d_{out}\times d_{in}},\ x\in\mathbb{R}^{1\times d_{in}}.\tag{1}$$

这样只需增加最外层维度即可组成 batch。传统线性代数常用列向量：

$$y=Wx.\tag{2}$$

此时 batch 维度要放在最后。本作业的数学公式主要使用列向量，但 NumPy/PyTorch 采用行主序，普通矩阵乘法时要注意转置。正确标记轴的 einsum 可避免这一问题。Matlab、Julia、Fortran 使用列主序；Python 生态沿用 C 的行主序。

### 3.3 基本构件：Linear 与 Embedding

#### 3.3.1 参数初始化

错误初始化可能导致梯度消失或爆炸。Pre-norm Transformer 相对鲁棒，但初始化仍影响训练速度和收敛。本作业使用：

- Linear：均值 0、方差 $2/(d_{in}+d_{out})$ 的正态分布，截断到 $[-3\sigma,3\sigma]$；
- Embedding：均值 0、方差 1 的正态分布，截断到 $[-3,3]$；
- RMSNorm：全 1。

使用 `torch.nn.init.trunc_normal_` 完成截断正态初始化。

#### 3.3.2 Linear

实现无 bias 的线性变换：

$$y=Wx.\tag{3}$$

**题目（linear）：实现 Linear（1 分）**

实现继承 `torch.nn.Module` 的 `Linear` 类，接口类似 `nn.Linear`，接收 `in_features`、`out_features`、`device`、`dtype`。必须调用父类构造函数，把 $W$ 而非 $W^\top$ 存为 `nn.Parameter`，不得使用 `nn.Linear` 或 `nn.functional.linear`，并按规定初始化。完成 `adapters.run_linear` 后运行 `uv run pytest -k test_linear`。

#### 3.3.3 Embedding

Embedding 把整数 token ID 映射到 $d_{model}$ 维空间。矩阵形状是 `(vocab_size, d_model)`，输入 token ID 形状是 `(batch_size, sequence_length)`。

**题目（embedding）：实现 Embedding（1 分）**

实现继承 `torch.nn.Module` 的 `Embedding` 类，接口类似 `nn.Embedding`，接收 `num_embeddings`、`embedding_dim`、`device`、`dtype`。矩阵应为 `nn.Parameter`，最后一维为 $d_{model}$；不得使用 `nn.Embedding` 或 `nn.functional.embedding`。完成 `adapters.run_embedding` 后运行 `uv run pytest -k test_embedding`。

### 3.4 Pre-Norm Transformer Block

每个 Block 有多头自注意力和逐位置前馈网络两个子层。原始 Transformer 在子层加残差后做归一化，称为 post-norm。后续工作发现，在每个子层输入处归一化，并在最后一个 Block 后额外归一化，可改善训练稳定性。这种 pre-norm 结构保留了从输入 embedding 到最终输出的干净残差流，已成为 GPT-3、LLaMA、PaLM 等模型的常见选择。

#### 3.4.1 RMSNorm

给定 $a\in\mathbb{R}^{d_{model}}$：

$$\operatorname{RMSNorm}(a_i)=\frac{a_i}{\operatorname{RMS}(a)}g_i,\tag{4}$$

$$\operatorname{RMS}(a)=\sqrt{\frac{1}{d_{model}}\sum_{i=1}^{d_{model}}a_i^2+\epsilon}.$$

$g_i$ 是可学习增益，$\epsilon$ 通常为 $10^{-5}$。平方前把输入提升为 `float32` 防止溢出，完成后转回原 dtype。

原文给出的前向方法结构为：

```python
in_dtype = x.dtype
x = x.to(torch.float32)
# 在此执行 RMSNorm
...
result = ...
return result.to(in_dtype)
```

**题目（rmsnorm）：实现 RMSNorm（1 分）**

实现 `torch.nn.Module`，接收 `d_model`、`eps`、`device`、`dtype`；输入输出形状均为 `(batch_size, sequence_length, d_model)`。完成 `adapters.run_rmsnorm` 后运行 `uv run pytest -k test_rmsnorm`。

#### 3.4.2 逐位置前馈网络

原始 Transformer 使用两次线性变换和 ReLU，内部维度通常为输入的四倍。现代 LLM 常改用激活函数与门控结合的 SwiGLU，并去掉 bias。

$$\operatorname{SiLU}(x)=x\sigma(x)=\frac{x}{1+e^{-x}}.\tag{5}$$

$$\operatorname{GLU}(x,W_1,W_2)=\sigma(W_1x)\odot W_2x.\tag{6}$$

$$\operatorname{FFN}(x)=W_2\left(\operatorname{SiLU}(W_1x)\odot W_3x\right).\tag{7}$$

其中 $W_1,W_3\in\mathbb{R}^{d_{ff}\times d_{model}}$，$W_2\in\mathbb{R}^{d_{model}\times d_{ff}}$，通常 $d_{ff}=\frac83d_{model}$，实现时可舍入到接近的 64 倍数。Shazeer 的实验显示 SwiGLU 优于 ReLU 和无门控 SiLU；论文打趣称其成功“归因于神意”。

**题目（positionwise_feedforward）：实现逐位置前馈网络（2 分）**

实现由 SiLU 和 GLU 组成的 SwiGLU。允许使用 `torch.sigmoid` 保证数值稳定；$d_{ff}$ 约为 $\frac83d_{model}$ 且为 64 的倍数。完成 `adapters.run_swiglu` 后运行 `uv run pytest -k test_swiglu`。

#### 3.4.3 相对位置嵌入 RoPE

位置 $i$ 的 query 为 $q(i)=W_qx(i)\in\mathbb{R}^d$，应用旋转矩阵：

$$q'(i)=R_iq(i)=R_iW_qx(i).$$

第 $k$ 对维度按角度 $\theta_{i,k}=i/\Theta^{(2k-2)/d}$ 旋转：

$$R_i^k=\begin{pmatrix}\cos\theta_{i,k}&\sin\theta_{i,k}\\-\sin\theta_{i,k}&\cos\theta_{i,k}\end{pmatrix}.\tag{8}$$

完整 $R_i$ 是由这些 $2\times2$ 块组成的分块对角矩阵（式 9）。高效实现不应显式构造完整 $d\times d$ 矩阵。sin/cos 可跨层和 batch 复用，并注册为非持久 buffer，而非参数。对 key 使用相同规则；RoPE 无可学习参数。

**题目（rope）：实现 RoPE（2 分）**

实现 `RotaryPositionalEmbedding`，构造接口为 `__init__(theta: float, d_k: int, max_seq_len: int, device=None)`；前向接口为 `forward(x: torch.Tensor, token_positions: torch.Tensor)`。输入形状为 `(..., seq_len, d_k)`，位置 Tensor 为 `(..., seq_len)`，输出与输入同形。用位置索引预计算的 sin/cos。完成 `adapters.run_rope` 后运行 `uv run pytest -k test_rope`。

#### 3.4.4 缩放点积注意力

$$\operatorname{softmax}(v)_i=\frac{e^{v_i}}{\sum_{j=1}^ne^{v_j}}.\tag{10}$$

大值可能使指数溢出。softmax 对所有输入同时加减常数不变，因此先减去目标维度最大值，使最大元素变为 0。

**题目（softmax）：实现 softmax（1 分）**

实现对指定维度做 softmax 的函数，输出形状不变，必须使用减最大值保证稳定。完成 `adapters.run_softmax` 后运行 `uv run pytest -k test_softmax_matches_pytorch`。

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.\tag{11}$$

$Q\in\mathbb{R}^{n\times d_k}$、$K\in\mathbb{R}^{m\times d_k}$、$V\in\mathbb{R}^{m\times d_v}$ 都是输入，不是可学习参数。

mask 形状为 $\{True,False\}^{n\times m}$。`True` 表示 query 可以关注对应 key，`False` 表示不能。计算时，在 softmax 前给 `False` 位置加 $-\infty$。

**题目（scaled_dot_product_attention）：实现缩放点积注意力（5 分）**

query/key 形状为 `(batch_size, ..., seq_len, d_k)`，value 为 `(batch_size, ..., seq_len, d_v)`，输出为 `(batch_size, ..., seq_len, d_v)`。支持 `(seq_len, seq_len)` 的可选布尔 mask；允许位置的概率总和为 1，禁止位置概率为 0。完成 `adapters.run_scaled_dot_product_attention`，运行三维和四维输入测试。

#### 3.4.5 因果多头自注意力

$$\operatorname{MultiHead}(Q,K,V)=\operatorname{Concat}(head_1,\ldots,head_h),\tag{12}$$

$$head_i=\operatorname{Attention}(Q_i,K_i,V_i).\tag{13}$$

$$\operatorname{MultiHeadSelfAttention}(x)=W_O\operatorname{MultiHead}(W_Qx,W_Kx,W_Vx).\tag{14}$$

其中 $W_Q,W_K\in\mathbb{R}^{hd_k\times d_{model}}$、$W_V\in\mathbb{R}^{hd_v\times d_{model}}$、$W_O\in\mathbb{R}^{d_{model}\times hd_v}$。正常实现以三次矩阵乘法计算 Q、K、V；扩展目标是把三者合成一次投影。

因果 mask 使位置 $i$ 只能关注 $j\le i$，避免未来 token 泄漏真实答案。RoPE 只施加到 query 和 key，不施加到 value；head 维作为 batch 维，每个 head 使用相同位置旋转。

**题目（multihead_self_attention）：实现因果多头自注意力（5 分）**

实现 `torch.nn.Module`，至少接收 `d_model` 和 `num_heads`，并令 $d_k=d_v=d_{model}/h$。完成 `adapters.run_multihead_self_attention` 后运行对应测试。

### 3.5 完整 Transformer LM

每个 Block 有 MHA 和 SwiGLU 两个子层；每个子层都先 RMSNorm，再执行主操作，最后加残差。例如第一子层：

$$y=x+\operatorname{MultiHeadSelfAttention}(\operatorname{RMSNorm}(x)).\tag{15}$$

**题目（transformer_block）：实现 Transformer Block（3 分）**

实现图 2 的 pre-norm Block，至少接收 `d_model`、`num_heads`、`d_ff`。完成 `adapters.run_transformer_block`，运行 `uv run pytest -k test_transformer_block`。交付物：通过测试的 Block。

**题目（transformer_lm）：实现 Transformer LM（3 分）**

把 embedding、`num_layers` 个 Block、最终 norm 和 LM head 组装起来。除 Block 参数外，还需 `vocab_size`、`context_length`、`num_layers`。完成 `adapters.run_transformer_lm` 并通过对应测试。

#### 资源核算

Transformer 大多数 FLOPs 来自矩阵乘法。若 $A\in\mathbb{R}^{m\times n}$、$B\in\mathbb{R}^{n\times p}$，则 $AB$ 需要 $2mnp$ FLOPs：每个元素含 $n$ 次乘法和 $n$ 次加法，共有 $mp$ 个元素。

**题目（transformer_accounting）：Transformer LM 资源核算（5 分）**

1. 对配置 `vocab_size=50257`、`context_length=1024`、`num_layers=48`、`d_model=1600`、`num_heads=25`、`d_ff=4288` 的 GPT-2 XL 形状模型，计算可训练参数数目和 float32 参数加载内存。交付物：一到两句话。
2. 列出一次前向传播中的所有矩阵乘法并计算总 FLOPs。交付物：带说明的列表和总数。
3. 指出哪些部分消耗最多 FLOPs。交付物：一到两句话。
4. 对 GPT-2 small（12 层、768 维、12 头）、medium（24 层、1024 维、16 头）、large（36 层、1280 维、20 头）重复分析，报告各组件 FLOPs 占比及规模变化趋势。
5. 把 GPT-2 XL 上下文增至 16,384，说明总 FLOPs 和各组件相对贡献如何变化。交付物：一到两句话。

## 4 训练 Transformer LM

已有分词器和模型后，还需构建交叉熵损失、AdamW 优化器，以及负责数据、checkpoint 和训练的基础设施。

### 4.1 交叉熵损失

对长度 $m+1$ 的序列，模型在每个位置定义 $p_\theta(x_{i+1}\mid x_{1:i})$。训练集 $D$ 上的负对数似然为：

$$\ell(\theta;D)=\frac{1}{|D|m}\sum_{x\in D}\sum_{i=1}^{m}-\log p_\theta(x_{i+1}\mid x_{1:i}).\tag{16}$$

一次前向传播会同时给出全部位置的预测。若位置 $i$ 的 logits 为 $o_i\in\mathbb{R}^{vocab\_size}$：

$$p(x_{i+1}\mid x_{1:i})=\operatorname{softmax}(o_i)[x_{i+1}]=\frac{\exp(o_i[x_{i+1}])}{\sum_{a=1}^{vocab\_size}\exp(o_i[a])}.\tag{17}$$

**题目（cross_entropy）：实现交叉熵（1 分）**

实现从 logits 和 targets 计算 $-\log\operatorname{softmax}(o_i)[x_{i+1}]$ 的函数。要求减最大值、尽可能抵消 `log` 和 `exp`、支持任意额外 batch 维并对 batch 求平均。完成 `adapters.run_cross_entropy` 后运行 `uv run pytest -k test_cross_entropy`。

长度 $m$ 序列的困惑度为：

$$\operatorname{perplexity}=\exp\left(\frac1m\sum_{i=1}^m\ell_i\right).\tag{18}$$

### 4.2 SGD 优化器

从随机参数 $\theta_0$ 开始，每步更新：

$$\theta_{t+1}\leftarrow\theta_t-\alpha_t\nabla L(\theta_t;B_t),\tag{19}$$

其中 $B_t$ 是从数据集采样的 batch，学习率和 batch size 是超参数。

实现优化器时继承 `torch.optim.Optimizer`。构造函数初始化参数和默认超参数，`step()` 在反向传播后原地更新每个拥有梯度的参数。PDF 给出一个学习率按 $\alpha/\sqrt{t+1}$ 衰减的 SGD 示例，并用平方损失演示典型训练循环：清梯度、计算损失、反向传播、优化器更新。

该变体的更新为：

$$\theta_{t+1}=\theta_t-\frac{\alpha}{\sqrt{t+1}}\nabla L(\theta_t;B_t).\tag{20}$$

原文示例代码如下：

```python
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss
```

构造函数把参数和默认超参数交给基类；若参数只是一个 `nn.Parameter` 集合，基类会创建单个参数组并设置默认值。`step()` 遍历参数组及参数，按式（20）更新，并为每个参数保存迭代数。可选 `closure` 用于重新计算损失，本作业中的优化器不会用到，但示例保留它以符合 API。

```python
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)
for t in range(100):
    opt.zero_grad()
    loss = (weights**2).mean()
    print(loss.cpu().item())
    loss.backward()
    opt.step()
```

训练语言模型时，参数来自 `model.parameters()`，损失在采样 batch 上计算，但循环的基本结构相同。

**题目（learning_rate_tuning）：调节学习率（1 分）**

在该 SGD 玩具例子中，分别用 $10^1$、$10^2$、$10^3$ 的学习率训练十步，观察损失是下降更快、更慢还是发散。交付物：一到两句话。

### 4.3 AdamW

现代 LM 通常使用 Adam 的变体。AdamW [23] 将 weight decay 与梯度更新解耦，每步把参数向 0 拉近。它为每个参数保存一阶和二阶矩估计，以额外内存换取稳定性和收敛速度。

除学习率 $\alpha$ 外，超参数还有 $(\beta_1,\beta_2)$、weight decay $\lambda$ 和稳定项 $\epsilon$。常见 $\beta$ 是 $(0.9,0.999)$，LLaMA、GPT-3 等大模型常用 $(0.9,0.95)$。

算法从 $m=0,v=0$ 开始。每步计算梯度 $g$，进行 bias correction：

$$\alpha_t=\alpha\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t},$$

先应用 $\theta\leftarrow\theta-\alpha\lambda\theta$，再更新：

$$m\leftarrow\beta_1m+(1-\beta_1)g,$$

$$v\leftarrow\beta_2v+(1-\beta_2)g^2,$$

$$\theta\leftarrow\theta-\alpha_t\frac{m}{\sqrt v+\epsilon}.$$

注意 $t$ 从 1 开始。

**题目（adamw）：实现 AdamW（2 分）**

继承 `torch.optim.Optimizer`，构造函数接收 $\alpha$、$\beta$、$\epsilon$、$\lambda$。可使用 `self.state` 为每个参数保存矩估计。完成 `adapters.get_adamw_cls` 后运行 `uv run pytest -k test_adamw`。

**题目（adamw_accounting）：AdamW 训练资源核算（2 分）**

假设所有 Tensor 均为 float32：

1. 计算 AdamW 的峰值内存，分别给出参数、激活、梯度和优化器状态的代数表达式及总和。表达式使用 `batch_size`、`vocab_size`、`context_length`、`num_layers`、`d_model`、`num_heads`，并令 $d_{ff}=\frac83d_{model}$。激活只考虑 Block 中的 RMSNorm、MHA 各步骤、SwiGLU 各步骤，以及最终 RMSNorm、输出 embedding、logits 交叉熵。
2. 代入 GPT-2 XL，得到仅依赖 `batch_size` 的 $a\cdot batch\_size+b$，并求 80 GB 内可容纳的最大 batch。
3. 一步 AdamW 需要多少 FLOPs？给出表达式及简短理由。
4. MFU 是实际 token 吞吐与硬件理论峰值 FLOPs 吞吐的比值。H100 的 TensorFloat-32 理论峰值为 495 TFLOP/s。假设 MFU 为 50%，反向传播 FLOPs 是前向的两倍，计算单张 H100 上用 batch 1024 训练 GPT-2 XL 400K 步所需小时数，并说明理由。

### 4.4 学习率调度

最合适的学习率会随训练变化。Transformer 常先使用较大更新，之后逐渐衰减。本作业实现 LLaMA 使用的带 warmup 的 cosine annealing。调度器根据当前步和相关参数返回该步学习率。

给定当前步 $t$、最大学习率 $\alpha_{max}$、最小学习率 $\alpha_{min}$、warmup 步数 $T_w$、余弦衰减结束步 $T_c$：

$$\alpha_t=\frac{t}{T_w}\alpha_{max},\qquad t<T_w;$$

$$\alpha_t=\alpha_{min}+\frac12\left(1+\cos\left(\frac{t-T_w}{T_c-T_w}\pi\right)\right)(\alpha_{max}-\alpha_{min}),\quad T_w\le t\le T_c;$$

$$\alpha_t=\alpha_{min},\qquad t>T_c.$$

有时也会让学习率重新上升（restart），以帮助越过局部最优。

**题目（learning_rate_schedule）：实现带 warmup 的余弦调度（1 分）**

实现接收上述五个参数并返回 $\alpha_t$ 的函数。完成 `adapters.get_lr_cosine_schedule` 后运行对应测试。

### 4.5 梯度裁剪

某些样本会产生过大梯度并破坏训练稳定性。设所有参数的梯度为 $g$，若 $\|g\|_2\le M$ 则不变；否则乘以：

$$\frac{M}{\|g\|_2+\epsilon},$$

其中 $\epsilon$ 通常为 $10^{-6}$，裁剪后范数会略小于 $M$。

**题目（gradient_clipping）：实现梯度裁剪（1 分）**

函数接收参数列表和最大 $\ell_2$ 范数，原地修改每个参数梯度，使用 $\epsilon=10^{-6}$。完成 `adapters.run_gradient_clipping` 后运行对应测试。

## 5 训练循环

现在把 tokenized data、模型和优化器组合起来。

### 5.1 Data Loader

分词后的数据是一个 token 序列 $x=(x_1,\ldots,x_n)$。即使原始数据由独立文档组成，通常也会用 `<|endoftext|>` 等分隔符将它们拼接成单个序列。

Data loader 产生 batch。每个 batch 含 $B$ 个长度为 $m$ 的输入序列及对应的长度 $m$ 的 next-token targets。例如 $B=1,m=3$ 时，`[x2,x3,x4]` 与 `[x3,x4,x5]` 是一组可能样本。

这种形式使任意 $1\le i\le n-m$ 都能成为合法起点；所有序列等长，无需 padding，提高硬件利用率；也无需把完整数据集载入内存。

**题目（data_loading）：实现数据加载（2 分）**

函数接收 token ID NumPy 数组、`batch_size`、`context_length` 和设备字符串，返回输入与 next-token targets。二者形状均为 `(batch_size, context_length)`，且位于指定设备。完成 `adapters.run_get_batch` 后运行 `uv run pytest -k test_get_batch`。

低资源训练时，CPU 使用 `cpu`，Apple Silicon 使用 `mps`。若数据过大，可用 `mmap` 将磁盘文件映射为虚拟内存，按需加载。NumPy 可使用 `np.memmap`，或对 `np.load` 指定 `mmap_mode='r'`。加载时指定正确 dtype，并检查数值未超过预期词表大小。

### 5.2 Checkpoint

训练可能因超时或机器故障中断，也可能需要事后研究中间模型。因此 checkpoint 必须保存恢复训练所需的全部状态：模型权重、AdamW 等有状态优化器的状态，以及恢复学习率调度所需的迭代数。

PyTorch 的模型与优化器都提供 `state_dict()` 和 `load_state_dict()`；`torch.save` 和 `torch.load` 可序列化包含 Tensor 与普通 Python 对象的结构。

**题目（checkpointing）：实现模型 checkpoint（1 分）**

实现以下两个函数：

```text
save_checkpoint(model, optimizer, iteration, out)
load_checkpoint(src, model, optimizer)
```

保存函数将模型状态、优化器状态和 iteration 写入路径或二进制文件对象；参数类型分别是 `torch.nn.Module`、`torch.optim.Optimizer`、`int`，以及路径或二进制 IO。加载函数从路径或文件对象恢复模型、优化器并返回保存的 iteration。完成 `adapters.run_save_checkpoint`、`adapters.run_load_checkpoint` 后运行 `uv run pytest -k test_checkpointing`。

### 5.3 训练循环

把已实现组件组装进主训练脚本。后续需要反复改变超参数，因此应容易通过命令行等方式启动不同配置。

**题目（training_together）：组合全部组件（4 分）**

交付物：编写使用用户输入训练模型的训练脚本，至少支持：

- 配置和控制模型与优化器超参数；
- 使用 `np.memmap` 节省内存地加载大型训练集和验证集；
- 将 checkpoint 保存到用户指定路径；
- 定期记录训练和验证表现，可输出到控制台或 Weights & Biases 等服务。

## 6 生成文本

语言模型接收长度为 `sequence_length` 的整数序列，输出 `(sequence_length, vocab_size)` 矩阵，每个位置给出下一个 token 的预测分布。最后还需要把它变成新序列的采样过程。

模型最后一层输出 logits，需要先用式（10）的 softmax 转成概率。

给定 prompt $x_{1:t}$，取模型输出矩阵最后一个位置的向量 $v$，并采样：

$$P(x_{t+1}=i\mid x_{1:t})=\frac{\exp(v_i)}{\sum_j\exp(v_j)},\tag{21}$$

$$v=\operatorname{TransformerLM}(x_{1:t})_t\in\mathbb{R}^{vocab\_size}.\tag{22}$$

每次把新 token 追加到下一步输入，直到生成 `<|endoftext|>` 或达到用户指定最大长度。

### 解码技巧

小模型生成质量可能较低。温度缩放使用参数 $\tau$：

$$\operatorname{softmax}(v,\tau)_i=\frac{\exp(v_i/\tau)}{\sum_{j=1}^{vocab\_size}\exp(v_j/\tau)}.\tag{23}$$

当 $\tau\to0$，最大 logit 占据主导，分布趋近集中在最大元素的 one-hot。

Nucleus/top-p sampling 会截断低概率 token。设 $q$ 是温度缩放后的分布，$V(p)$ 是满足 $\sum_{j\in V(p)}q_j\ge p$ 的最小索引集合，则：

$$P(x_{t+1}=i\mid q)=\begin{cases}\dfrac{q_i}{\sum_{j\in V(p)}q_j},&i\in V(p),\\0,&\text{其他。}\end{cases}\tag{24}$$

可按概率降序排序，依次选择最大项，直到累计概率达到 $p$。

**题目（decoding）：解码（3 分）**

实现语言模型解码函数，建议支持：

- 对用户 prompt 生成补全，直到 `<|endoftext|>`；
- 控制最大生成 token 数；
- 按给定温度缩放 next-token 分布；
- 按用户阈值执行 top-p/nucleus sampling。

## 7 实验

现在组合所有内容，在预训练数据集上训练小型语言模型。

### 7.1 实验运行与交付

理解 Transformer 组件动机的最好方式，是亲自修改并运行。实验必须快速、可复现并有记录。作业使用约 17M 参数的小模型和 TinyStories，系统地消融组件和改变超参数，并要求提交实验日志及每项实验的学习曲线。

为了提交 loss 曲线，应定期计算验证 loss，同时记录 step 数和 wall-clock time。可以使用 Weights & Biases 等日志工具。

**题目（experiment_log）：实验日志（3 分）**

为训练和评估代码建立实验追踪基础设施，能够按梯度步数和 wall-clock time 记录实验与 loss 曲线。交付物：日志基础设施代码，以及记录本节所有尝试的实验文档。

### 7.2 TinyStories

TinyStories [1] 是简单且训练快速的数据集。原文示例翻译如下：

> 从前，有一个名叫 Ben 的小男孩。Ben 喜欢探索身边的世界。他见过许多令人惊叹的东西，比如商店里陈列的漂亮花瓶。一天，Ben 走过商店时，发现了一只非常特别的花瓶。看到它时，Ben 惊呆了！他说：“哇，那真是一只很棒的花瓶！我可以买下它吗？”店主笑着说：“当然可以。你可以把它带回家，让所有朋友看看它有多么惊人！”于是 Ben 把花瓶带回家，并为它感到无比自豪！他叫来朋友们，向他们展示这只令人惊叹的花瓶。所有朋友都觉得它非常漂亮，简直不敢相信 Ben 这么幸运。这就是 Ben 如何在商店中发现一只神奇花瓶的故事！

#### 7.2.1 超参数调节

起始建议：

- 词表大小：10,000。典型词表为数万到数十万，可改变它并观察词表和模型行为；
- 上下文长度：256。TinyStories 不需要很长序列，但 OpenWebText 可能需要；观察每步时间和最终困惑度；
- `d_model=512`；
- `d_ff=1344`，约为 $\frac83d_{model}$ 且是 64 的倍数；
- RoPE $\Theta=10000$；
- 4 层、16 个 head，约 17M 个非 embedding 参数；
- 总处理 token：327,680,000，即 batch size × 总步数 × context length 应约等于该值。

还需通过试验确定学习率、warmup、AdamW 的 $\beta_1,\beta_2,\epsilon$ 和 weight decay。

#### 7.2.2 组合训练

使用训练好的 BPE 编码数据，再运行训练循环。正确且高效的实现按上述参数在一张 B200 上约需 20 到 30 分钟。若显著更慢，检查 dataloader、checkpoint、验证 loss 是否成为瓶颈，以及模型是否正确 batch 化。

#### 7.2.3 调试模型架构

建议熟悉 IDE 调试器而非依赖打印；文本编辑器用户可用 `ipdb`。其他实践包括：

- 先让模型过拟合单个 minibatch，正确实现应能迅速把 loss 降至接近 0；
- 在组件内设置断点，检查中间 Tensor 形状；
- 监控激活、权重和梯度范数，确认没有爆炸或消失。

**题目（learning_rate）：调节学习率（2 B200 小时，3 分）**

1. 对学习率做 sweep，报告最终 loss，发散则注明。交付物：多条学习曲线、搜索策略说明，以及 TinyStories 每 token 验证 loss 不高于 1.45 的模型。
2. 经验说最佳学习率位于“稳定性边缘”。研究学习率开始发散的位置与最佳学习率的关系。交付物：递增学习率的曲线，其中至少一个 run 发散，并分析与收敛速度的关系。

低资源配置：CPU/MPS 可把总 token 降至 40,000,000，并将目标验证 loss 放宽到 2.00。课程组在 36 GB M4 Max 上使用 $32\times5000\times256=40,960,000$ token，CPU 约 1 小时 22 分，MPS 约 36 分钟，5000 步验证 loss 为 1.80。余弦衰减应恰在第 $N$ 步达到最小学习率。MPS 不要启用 TF32；PyTorch 2.9.0 下课程组观察到某些内核会静默产生错误并导致训练不稳定。CPU 可使用 `torch.compile`，MPS 可用 `aot_eager` 优化反向传播；MPS 不支持 Inductor。

**题目（batch_size_experiment）：改变 batch size（1 B200 小时，1 分）**

从 1 一直试到 GPU 内存上限，并包含 64、128 等典型中间值。交付物：不同 batch size 的学习曲线，必要时重新优化学习率，并用几句话讨论 batch size 对训练的影响。

**题目（generate）：生成文本（1 分）**

使用 decoder 和训练 checkpoint 报告生成文本，可调 temperature、top-p 等参数。交付物：至少 256 token 的文本（或在首个 `<|endoftext|>` 停止），简评流畅度，并指出至少两个影响质量的因素。

课程给出的参考生成样例翻译如下：

> 从前，有一个名叫 Lily 的漂亮女孩。她喜欢吃口香糖，尤其喜欢那颗黑色的大口香糖。一天，Lily 的妈妈请她帮忙做晚饭。Lily 非常兴奋！她喜欢帮助妈妈。Lily 的妈妈为晚饭煮了一大锅汤。Lily 非常开心，说：“谢谢你，妈妈！我爱你。”她帮妈妈把汤倒进一个大碗。晚饭后，Lily 的妈妈又做了一些美味的汤。Lily 很喜欢！她说：“谢谢你，妈妈！这汤真好喝！”妈妈笑着说：“Lily，我很高兴你喜欢。”她们做完饭，又继续一起烹饪。故事结束。

低资源配置处理 40M token 后，文本仍像英语，但不如上例流畅。课程给出的样例翻译如下：

> 从前，有一个名叫 Sue 的小女孩。Sue 有一颗她非常喜欢的牙齿。那是他最好的头。一天，Sue 出门散步，遇到了一只瓢虫！他们成了好朋友，并一起在小路上玩耍。
>
> “嘿，Polly！我们出去吧！”Tim 说道。Sue 看着天空，发现很难找到一种闪闪发光地跳舞的方法。她笑着同意帮助那个说话的东西！
>
> 当 Sue 看着天空移动时，那是什么。她……

### 7.3 消融与架构修改

#### 消融 1：归一化

**题目（layer_norm_ablation）：移除 RMSNorm 并训练（0.5 B200 小时，1 分）**

移除所有 RMSNorm，在此前最佳学习率下训练，观察结果，并尝试更低学习率是否能恢复稳定。交付物：移除 RMSNorm 后的曲线、其最佳学习率曲线，以及对 RMSNorm 影响的简短评论。

Pre-norm 定义为：

$$z=x+\operatorname{MultiHeadSelfAttention}(\operatorname{RMSNorm}(x)),\tag{25}$$

$$y=z+\operatorname{FFN}(\operatorname{RMSNorm}(z)).\tag{26}$$

原始 post-norm 为：

$$z=\operatorname{RMSNorm}(x+\operatorname{MultiHeadSelfAttention}(x)),\tag{27}$$

$$y=\operatorname{RMSNorm}(z+\operatorname{FFN}(z)).\tag{28}$$

**题目（pre_norm_ablation）：实现 post-norm 并训练（0.5 B200 小时，1 分）**

把 pre-norm 改为 post-norm 并训练。交付物：post-norm 与 pre-norm 的学习曲线对比。实验用于说明归一化及其位置都会显著影响 Transformer。

#### 消融 2：位置嵌入

decoder-only Transformer 的因果 mask 理论上可能使其在没有显式位置嵌入时推断相对或绝对位置 [28, 29]。

**题目（no_pos_emb）：实现 NoPE（0.5 B200 小时，1 分）**

从 RoPE 模型中完全移除位置信息并训练。交付物：RoPE 与 NoPE 的学习曲线比较。

#### 消融 3：SwiGLU 与 SiLU

无门控 SiLU 前馈网络为：

$$\operatorname{FFN}_{SiLU}(x)=W_2\operatorname{SiLU}(W_1x).\tag{29}$$

SwiGLU 使用约 $\frac83d_{model}$ 的内部维度和三个权重矩阵；为了近似匹配参数量，无门控 SiLU 基线应使用 $d_{ff}=4d_{model}$。

**题目（swiglu_ablation）：SwiGLU 与 SiLU（0.5 B200 小时，1 分）**

交付物：参数量大致匹配时的两条学习曲线，以及对结果的简短讨论。

低资源学习者可继续在 TinyStories 上测试修改，并以验证 loss 为指标；OpenWebText 训练到流畅文本需要很长时间。

### 7.4 在 OpenWebText 上运行

OpenWebText [2] 是从网络抓取构建的、更标准且更复杂多样的预训练数据集。作业提供单个文本文件样本。应先浏览数据，了解网页抓取语料的样子。原文示例翻译如下：

> Baseball Prospectus 的技术主管 Harry Pavlidis 在雇用 Jonathan Judge 时冒了一个险。
>
> Pavlidis 知道，正如 Alan Schwarz 在《The Numbers Game》中所写：“美国文化的任何角落，都没有棒球运动员的表现那样被如此精确地统计、如此热情地量化。”只需点击几下，你就能知道 Noah Syndergaard 的快速球在飞向本垒板的途中每分钟旋转超过 2,100 圈，知道 Nelson Cruz 在 2016 年符合资格的击球手中拥有最高的平均击球初速度，还能发现无数仿佛来自电子游戏或科幻小说的零碎数据。不断上涨的数据海洋赋予棒球文化中的一个角色越来越大的力量：业余数据分析爱好者。
>
> 这种赋能也带来了更严格的审视，不仅针对测量结果，也针对测量背后的人和出版机构。在 Baseball Prospectus 工作的 Pavlidis 非常了解定量分析不完美时会引发的反弹。他也知道网站的接球指标需要重新设计，而且需要一位学识深厚、能够解决复杂统计建模问题的人来完成这项工作。
>
> “他让我们大吃一惊。”——Harry Pavlidis
>
> 根据 Judge 的文章，以及他们在一次网站组织的球场活动中的交流，Pavlidis 直觉认为 Judge “真正懂这件事”。……

本实验可能需要重新调节学习率、batch size 等超参数。

**题目（main_experiment）：OpenWebText 实验（2 B200 小时，2 分）**

使用与 TinyStories 相同的模型架构和总迭代数，在 OpenWebText 上训练。

交付物：

- OpenWebText 学习曲线，并说明与 TinyStories loss 的差别及其含义；
- 与 TinyStories 相同格式的生成文本，评价流畅度，并解释为何模型和计算预算相同，输出质量却更差。

### 7.5 自定义修改与排行榜

最后尝试改进 Transformer 架构，并与其他学生比较超参数和架构。

排行榜只有两条限制：

- 运行时间：一张 B200 上最多 45 分钟；若使用 SLURM 或 Modal，可在提交脚本中强制限制；
- 数据：只能使用课程提供的 OpenWebText 训练集。

除此之外没有限制。可参考 Llama 3、Qwen 2.5 等开源模型，或 modded-nanogpt speedrun 仓库中的小规模预训练优化。例如输入与输出 embedding 权重共享；若尝试权重共享，可能需要减小 embedding/LM head 初始化标准差。

完整 45 分钟运行前，应先在 OpenWebText 子集或 TinyStories 上测试。排行榜中有效的修改不一定能泛化到大规模预训练，课程将在 scaling laws 单元进一步讨论。

**题目（leaderboard）：排行榜（10 B200 小时，6 分）**

按上述规则训练，在 0.75 B200 小时内最小化验证 loss。交付物：最终验证 loss、一条以 wall-clock time 为横轴且清楚显示少于 45 分钟的学习曲线，以及修改说明。提交至少应优于验证 loss 5.0 的朴素基线。提交地址：<https://github.com/stanford-cs336/assignment1-basics-leaderboard>。

## 参考文献

1. R. Eldan and Y. Li, “TinyStories: How Small Can Language Models Be and Still Speak Coherent English?,” 2023.
2. A. Gokaslan, V. Cohen, E. Pavlick, and S. Tellex, “OpenWebText corpus,” 2019.
3. R. Sennrich, B. Haddow, and A. Birch, “Neural Machine Translation of Rare Words with Subword Units,” ACL, 2016.
4. C. Wang, K. Cho, and J. Gu, “Neural Machine Translation with Byte-Level Subwords,” 2019.
5. P. Gage, “A new algorithm for data compression,” C Users Journal, 1994.
6. A. Radford et al., “Language Models are Unsupervised Multitask Learners,” 2019.
7. A. Radford et al., “Improving Language Understanding by Generative Pre-Training,” 2018.
8. A. Vaswani et al., “Attention is All You Need,” NeurIPS, 2017.
9. T. Q. Nguyen and J. Salazar, “Transformers without Tears: Improving the Normalization of Self-Attention,” IWSLT, 2019.
10. R. Xiong et al., “On Layer Normalization in the Transformer Architecture,” ICML, 2020.
11. J. L. Ba, R. Kiros, and G. Hinton, “Layer Normalization,” 2016.
12. H. Touvron et al., “LLaMA: Open and Efficient Foundation Language Models,” 2023.
13. B. Zhang and R. Sennrich, “Root Mean Square Layer Normalization,” NeurIPS, 2019.
14. A. Grattafiori et al., “The Llama 3 Herd of Models,” <https://arxiv.org/abs/2407.21783>.
15. A. Yang et al., “Qwen2.5 Technical Report,” arXiv:2412.15115, 2024.
16. A. Chowdhery et al., “PaLM: Scaling Language Modeling with Pathways,” 2022.
17. D. Hendrycks and K. Gimpel, “Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units,” 2016.
18. S. Elfwing, E. Uchibe, and K. Doya, “Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning,” <https://arxiv.org/abs/1702.03118>.
19. Y. N. Dauphin et al., “Language Modeling with Gated Convolutional Networks,” <https://arxiv.org/abs/1612.08083>.
20. N. Shazeer, “GLU Variants Improve Transformer,” 2020.
21. J. Su et al., “RoFormer: Enhanced Transformer with Rotary Position Embedding,” 2021.
22. D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” ICLR, 2015.
23. I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” ICLR, 2019.
24. T. B. Brown et al., “Language Models are Few-Shot Learners,” NeurIPS, 2020.
25. J. Kaplan et al., “Scaling Laws for Neural Language Models,” 2020.
26. J. Hoffmann et al., “Training Compute-Optimal Large Language Models,” 2022.
27. A. Holtzman et al., “The Curious Case of Neural Text Degeneration,” ICLR, 2020.
28. Y.-H. H. Tsai et al., “Transformer Dissection: An Unified Understanding for Transformer's Attention via the Lens of Kernel,” EMNLP-IJCNLP, 2019.
29. A. Kazemnejad et al., “The Impact of Positional Encoding on Length Generalization in Transformers,” NeurIPS, 2023.
