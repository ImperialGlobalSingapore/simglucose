#!/usr/bin/env python
# Test script for Chronos time series model encoding

import torch
from transformers import AutoModel
from chronos.chronos import ChronosConfig, ChronosModel

def test_chronos_encoding():
    # Create the same ChronosConfig as in your script
    chronos_config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs={"low_limit": -15.0, "high_limit": 15.0},
        n_tokens=4096,
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=True,
        model_type="seq2seq",
        context_length=1440,
        prediction_length=64,
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    
    # Load the underlying model (using the same model as in your script)
    base_model = AutoModel.from_pretrained("amazon/chronos-t5-small")
    
    # Create the ChronosModel instance
    ts_model = ChronosModel(chronos_config, base_model)
    
    # Create a sample time series (dummy blood glucose data)
    # Simulating a sequence of blood glucose readings
    sample_time_series = torch.tensor([120.0]*1440, 
                                     dtype=torch.float32)
    batch_time_series = torch.stack([sample_time_series]*8)
    # Create a batch (add batch dimension)
    # batch_time_series = sample_time_series.unsqueeze(0)  # Shape: [1, sequence_length]
    
    print(f"Input time series shape: {batch_time_series.shape}")
    
    # Create the tokenizer from config
    chronos_tokenizer = chronos_config.create_tokenizer()
    
    # Tokenize the time series data first
    token_ids, attention_mask, *_ = chronos_tokenizer.context_input_transform(batch_time_series)
    
    print(f"Tokenized time series shape: {token_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # Test the encode method
    encoded_ts = ts_model.encode(token_ids, attention_mask)
    
    print(f"Encoded time series output shape: {encoded_ts.shape}")
    print(f"Encoded time series sample values:\n{encoded_ts[0, :5, :5]}")  # Print first 5x5 values
    
    return encoded_ts

if __name__ == "__main__":
    print("Testing Chronos time series encoding...")
    encoded_output = test_chronos_encoding()
    print("Encoding test completed successfully!")