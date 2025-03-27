# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Full training
python train_llama.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name ./glucose_questions_answers.jsonl \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir Qwen2-0.5B-SFT

# LoRA
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 100 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
"""

import argparse
import json
import os
from datasets import Dataset
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
from qwen_ts_for_inference import Qwen2ForCausalLM
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

import chronos
from chronos import ChronosConfig, ChronosModel, ChronosTokenizer

from accelerate.utils import DistributedDataParallelKwargs


def main(script_args, training_args, model_args):
    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # Create model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

    # if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
    #     from transformers import AutoModelForImageTextToText

    #     model_kwargs.pop("use_cache", None)  # Image models do not support cache
    #     model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    # else:
    #     model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)


    ################
    # Dataset
    ################
    # Load the patient data JSON file
    data_path = script_args.dataset_name

    chronos_config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",  # From YAML "MeanScaleUniformBins"
        tokenizer_kwargs={"low_limit": -15.0, "high_limit": 15.0},  # From YAML
        n_tokens=4096,  # From YAML
        n_special_tokens=2,  # From second snippet (not in YAML)
        pad_token_id=0,  # From second snippet (not in YAML)
        eos_token_id=1,  # From second snippet (not in YAML)
        use_eos_token=True,  # From YAML
        model_type="seq2seq",  # From YAML
        context_length=512,  # From YAML
        prediction_length=64,  # From YAML
        num_samples=20,  # From YAML
        temperature=1.0,  # From second snippet (not in YAML)
        top_k=50,  # From second snippet (not in YAML)
        top_p=1.0,  # From second snippet (not in YAML)
    )
    ts_model = ChronosModel(chronos_config, AutoModel.from_pretrained("amazon/chronos-t5-small"))
    # model.model.ts_embeddings = ts_model
    chronos_tokenizer = chronos_config.create_tokenizer()
    model = Qwen2ForCausalLM.from_pretrained(model_args.model_name_or_path)
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not os.path.exists('data/train/'):
        with open(data_path, "r") as f:
            patient_data = [json.loads(s) for s in f.read().splitlines()]
        # Create a dataset with question-answer pairs from all patients
        formatted_data = []
        for patient in tqdm(patient_data, desc='Preparing QA and timeseries data'):
            # Extract QA pairs from each patient record
            qa_pairs = patient["questions_and_answers"]
            time_series = patient['patient_data']['bg_mgdl']
            for qa in qa_pairs:
                timeseries_token_ids, attn_mask, *_ = chronos_tokenizer.context_input_transform(torch.Tensor(time_series).unsqueeze(0))
                # print(timeseries_token_ids)
                formatted_data.append(
                    {
                        "prompt": [{"role": "user", "content": qa["question"]}],
                        "completion": [{"role": "assistant", "content": f"{qa['explanation']}\nThe answer is {qa['answer']}"}],
                        "timeseries_input_ids": timeseries_token_ids.squeeze().tolist(),
                        "timeseries_attention_mask": attn_mask.squeeze().tolist()
                    }
                )
        # Split the dataset for training and testing
        # Default split: 90% train, 10% test
        dataset_size = len(formatted_data)
        train_size = int(0.9 * dataset_size)

        train_data = formatted_data[:train_size]
        test_data = formatted_data[train_size:]

        # Create HuggingFace datasets
        train_dataset = Dataset.from_list(train_data)
        test_dataset = Dataset.from_list(test_data)
        
        train_dataset.save_to_disk('data/train/')
        test_dataset.save_to_disk('data/test/')
    else:
        train_dataset = Dataset.load_from_disk('data/train/')
        test_dataset = Dataset.load_from_disk('data/test/')
    ################
    # Training
    ################
    # print(training_args.accelerator_config.)
    # assert False
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=os.path.basename(script_args.dataset_name))


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
