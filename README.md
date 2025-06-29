# Engineering Honours Research Project

## AI-driven Job Interview Feedback System

This project is designed to give users personalised detailed feedback on their behavioural interview performance. It uses a combination of multimodal AI models to analyse user responses, including audio, video, and text transcription analysis.

More information can be found in the [project proposal](https://part4project.foe.auckland.ac.nz/home/project/detail/5673/).

## Developers

- [Alex Liang](https://github.com/alux444)
- [Tony Lim](https://github.com/tonylxm)

### Supervisors

- [Dr. David Huang](https://profiles.auckland.ac.nz/david-huang)
- [Dr. Andrew Meads](https://profiles.auckland.ac.nz/andrew-meads)
- [Dr. Yu-Cheng Tu](https://profiles.auckland.ac.nz/yu-cheng-tu)

## Getting Started

To run the project, you will need to set up a Python virtual environment and install the required dependencies.

### Prerequisites

- [Python 3.11 or later](https://www.python.org/downloads/)
- [Docker](https://www.docker.com/products/docker-desktop) for running the backend and models
- [Node.js](https://nodejs.org/en/download/) for the frontend React app
- [PyTorch](https://download.pytorch.org/whl/cpu/torch-2.3.0%2Bcpu-cp311-cp311-linux_x86_64.whl) for local model inference

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/alux444/p4p
   cd p4p
   ```

2. Set up a Python virtual environment:

   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Make sure to download the local PyTorch wheel file from [here](https://download.pytorch.org/whl/cpu/torch-2.3.0%2Bcpu-cp311-cp311-linux_x86_64.whl) and place it in the root directory of the project. Do not change the filename.

4. In `backend`, create a `.env` file with the required environment variables. You can use the provided `.env.example` as a template.

   ```bash
   cp backend/.env.example backend/.env
   ```

   Fill in the `.env` file with your Azure Cognitive Services keys and endpoints.

5. From the root directory, run the Docker container:
   ```bash
   docker compose up --build
   ```

## File structure

```
p4p/
├── README.md                 # This README file
├── audio-analysis            # Audio analysis with fluency, confidence and emotion detection
├── backend                   # Simple backend for connecting to our cloud models
├── expression-recognition/   # Attempts with expression recognition libraries
│   ├── deepface/             # Deepface library
│   │   ├── live              # Example app with live recording
│   │   └── recording         # Example app with video file parsing
│   └── fer/                  # Fer library
│       └── live              # Example app with live recording
├── agent-prompts             # Initial agent prompts for each Azure model
├── frontend                  # React webapp for prompting questions + recording responses
├── media                     # Relevant media files
├── sentiment-analysis        # Sentiment analysis models
├── transcriber               # Transcribing app using OpenAI Whisper
└── torch.whl                 # Local PyTorch wheel file (torch-2.3.0+cpu-cp311-cp311-linux_x86_64.whl)
```
