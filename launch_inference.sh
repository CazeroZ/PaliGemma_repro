#!/bin/bash

MODEL_PATH="/home/cazero/ML/code_from_scratch/VLM/paligemma-3b-pt-224"
PROMPT="the famous building in Beijing, China, "
IMAGE_FILE_PATH="test_images/tiantan.jpg"
MAX_TOKENS_TO_GENERATE=200
TEMPERATURE=0.6
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="False"

python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \
