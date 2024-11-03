# PaLI-GEMMA Model Reproduction

A reproduction of **Google's PaLI-GEMMA** (an open vision-language model), mainly for self-learning purposes. This project references [Google's official code](https://github.com/google-research/big_vision) and the associated [research paper](https://arxiv.org/abs/2407.07726). It includes detailed notes on tensor shape transformations within the network for deeper understanding.

### Key Points
- **Focus**: Documenting tensor shape transformations at each layer for in-depth learning of the model structure.
- **Goal**: To provide a clear, readable version of PaLI-GEMMA for self-study.



### Setup

1. **Download Model Weights**: Download the model weights from [PaLI-GEMMA 3B on Hugging Face](https://huggingface.co/google/paligemma-3b-pt-224) 
   
2. **Clone the repository** and install dependencies:
   ```bash
   git clone git@github.com:CazeroZ/PaliGemma_repro.git
   cd PaliGemma_repro
   pip install -r requirements.txt
   ```

3. **Configure and Run the Inference Script**:
   - Open `launch_inference.sh` and modify the following variables as needed:
     - `MODEL_PATH`: Set this to the directory where the downloaded model weights are saved.
     - `PROMPT`: Update with the prompt you want to use for inference.
     - `IMAGE_FILE_PATH`: Set this to the path of the input image.


4. **Run the Inference**:
   ```bash
   sh launch_inference.sh
   ```
