# Embedding Pipeline & ONNX runtime

This document describes the embedding provider design, model details, and vector preprocessing stages in ZionDB.

---

## Selected Model: `all-MiniLM-L6-v2`

ZionDB uses `sentence-transformers/all-MiniLM-L6-v2` for Version 1.

- **Parameters**: 22 Million (extremely lightweight, fast CPU inference)
- **Output Dimensions**: 384
- **Context Window**: 256 tokens (truncated if longer)
- **ONNX Format**: Supported natively on Hugging Face Hub (no PyTorch dependency required).

---

## Embedding Provider Steps

When `embed(texts)` is called, the following steps execute sequentially:

```
[Input Texts]
     │
     ▼
[Tokenizer (Hugging Face AutoTokenizer)] -> Token IDs & Attention Mask
     │
     ▼
[ONNX Runtime Session run] -> Token Embeddings (shape: [batch, seq_len, 384])
     │
     ▼
[Mean Pooling] -> 1D Vector per text (shape: [batch, 384])
     │
     ▼
[L2 Normalization] -> Output Vectors (magnitude = 1.0)
```

### 1. Tokenization
We use `transformers.AutoTokenizer` to convert raw text strings into arrays of input IDs, attention masks, and (optionally) token type IDs.
We pad strings in the batch to match the length of the longest string and truncate any token list exceeding 256.

### 2. ONNX Inference
The token arrays are passed as input feeds to `onnxruntime.InferenceSession`. We filter out token type IDs if the session inputs do not require them.
The first output node yields the `last_hidden_state` (token-level embeddings).

### 3. Mean Pooling
The token embeddings include padding tokens. To retrieve a single sentence vector, we perform **mean pooling**:
1. Multiply token embeddings by the expanded attention mask (zeros out padding tokens).
2. Sum the vectors along the sequence dimension.
3. Divide by the token count (sum of the attention mask, clamped to $1e-9$ to prevent division by zero).

### 4. L2 Normalization
Finally, we compute the L2 norm for each sentence vector:

$$\hat{v} = \frac{v}{\max(\|v\|_2, 1e-9)}$$

This projects all vectors onto a unit hypersphere, making the dot product of two vectors identical to their cosine similarity.
