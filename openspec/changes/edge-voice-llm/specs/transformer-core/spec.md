## ADDED Requirements

### Requirement: Decoder-Only Modern Transformer Architecture
The system SHALL provide a pure PyTorch decoder-only Transformer module implementing RMSNorm, Rotary Position Embeddings (RoPE), Grouped-Query Attention (GQA), and SwiGLU Feed-Forward Networks.

#### Scenario: Forward pass computation
- **WHEN** a batch of input token IDs is passed to the transformer core with shape `(batch_size, seq_len)`
- **THEN** the model SHALL output logit tensors with shape `(batch_size, seq_len, vocab_size)` without error

#### Scenario: Grouped-Query Attention key-value head sharing
- **WHEN** the attention module is initialized with `n_heads` queries and `n_kv_heads` key/value projections where `n_heads > n_kv_heads`
- **THEN** key and value tensors SHALL be repeated to match query heads during scaled dot-product attention calculation

### Requirement: Autoregressive KV Caching for Generation
The system SHALL support an autoregressive KV cache mechanism that stores previous key and value projections to eliminate redundant sequence computation during token generation.

#### Scenario: Token generation with KV cache
- **WHEN** generating tokens one step at a time given a starting prompt
- **THEN** each incremental generation step SHALL only compute projections for the new token and concatenate to existing KV cache tensors

### Requirement: Byte-Pair Encoding Tokenizer
The system SHALL provide a Byte-Pair Encoding (BPE) tokenizer capable of encoding text strings into token IDs and decoding token IDs back to human-readable strings.

#### Scenario: Round-trip text tokenization
- **WHEN** an arbitrary valid UTF-8 string is encoded into token IDs and immediately decoded
- **THEN** the decoded output string SHALL match the input string exactly
