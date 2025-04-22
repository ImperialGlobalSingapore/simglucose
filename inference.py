import torch
import os
from transformers import AutoTokenizer
from qwen_ts_for_inference import Qwen2ForCausalLM
from chronos import ChronosConfig
from datasets import Dataset


def inference_with_multimodal_model(
    model,
    text_tokenizer: AutoTokenizer,
    chronos_tokenizer,
    example,
    max_new_tokens=100,
    do_sample=False,
    # temperature=1.2,
    # top_p=0.9
):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # Tokenize text input
    text_encoding = text_tokenizer(text_tokenizer.apply_chat_template(example['prompt'], tokenize=False, add_generation_prompt=True), return_tensors="pt")
    # print(text_tokenizer.decode(text_encoding.input_ids[0]))
    # assert False
    input_ids = text_encoding.input_ids.to(device)
    attention_mask = text_encoding.attention_mask.to(device)
    time_series_data = example['timeseries_input_ids']
    # Tokenize time series
    time_series_tensor = torch.tensor(time_series_data, dtype=torch.float32).unsqueeze(0)
    ts_input_ids, ts_attention_mask, *_ = chronos_tokenizer.context_input_transform(time_series_tensor)
    ts_input_ids = ts_input_ids.to(device)
    ts_attention_mask = ts_attention_mask.to(device)

    # Call generate
    with torch.no_grad():
        generated_ids = model.generate(
            # model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            timeseries_input_ids=ts_input_ids,
            timeseries_attention_mask=ts_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            # temperature=temperature,
            # top_p=top_p,
            use_cache=True,
        )

    return text_tokenizer.decode(generated_ids[0][len(input_ids[0]):], skip_special_tokens=True)


# Example usage
def sample_inference():
    # Specify the actual path to your local model
    # model_path = "Qwen2-0.5B-SFT"
    model_path = "/data/viktor/Qwen2-0.5b-SFT/"

    print(f"Loading model from: {model_path}")
    # Verify the path exists
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")

    # Load the model with local_files_only=True
    model = Qwen2ForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    print(model.generate.__module__)
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Load tokenizer
    text_tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True, use_fast=True
    )

    # Create Chronos tokenizer with the same config used during training
    chronos_config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs={"low_limit": -15.0, "high_limit": 15.0},
        n_tokens=4096,
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=True,
        model_type="seq2seq",
        context_length=512,
        prediction_length=64,
        num_samples=20,  # From YAML
        temperature=1,  # From second snippet (not in YAML)
        top_k=50,  # From second snippet (not in YAML)
        top_p=1.0,  # From second snippet (not in YAML)
    )
    chronos_tokenizer = chronos_config.create_tokenizer()

    test_dataset = Dataset.load_from_disk("data/test/")

    # Run inference
    try:
        print("Starting inference...")
        response = inference_with_multimodal_model(
            model,
            text_tokenizer,
            chronos_tokenizer,
            example=test_dataset[0],
            # text_input=question,
            # time_series_data=time_series_data,
            do_sample=False,
        )

        print(f"\nQuestion: {test_dataset[0]['prompt'][0]['content']}")
        print(f"Response: {response}")
        print(f"Actual response: {test_dataset[0]['completion'][0]['content']}")
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    sample_inference()
