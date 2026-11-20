## ADDED Requirements

### Requirement: Cross-Platform Training Loop
The system SHALL provide an end-to-end training pipeline supporting PyTorch automatic device selection (`cuda`, `mps`, or `cpu`) with AdamW optimization and cosine learning rate scheduling with warmup.

#### Scenario: Running training step on available device
- **WHEN** the training script is invoked with a training dataset and configuration
- **THEN** the training loop SHALL automatically utilize the optimal compute backend and log cross-entropy loss per step

### Requirement: Checkpoint Saving and Resumption
The system SHALL save model weights, optimizer states, and hyperparameter configurations into checkpoint files.

#### Scenario: Loading checkpoint for inference or continued training
- **WHEN** a saved checkpoint path is passed to the loader
- **THEN** the model SHALL restore identical weight parameters and produce reproducible output logits for identical input tokens
