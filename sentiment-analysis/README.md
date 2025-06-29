## To Run
1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Install PyTorch (choose the appropriate command based on your system):
   For CPU-only support, run:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```
   or for GPU support, follow the instructions at [PyTorch's official site](https://pytorch.org/get-started/locally/).
3. Run the FastAPI server:
   ```bash
    uvicorn main:app --reload --port 8001
    ```
4. Access the API documentation at:
   ```
   http://localhost:8001/docs
   ```
5. Use the `/sentiment-analysis` endpoint to analyse text. You can upload a file directly through the API documentation interface or send a POST request with text data.