# Vosk Speech Recognition Models

This directory contains Vosk models for speech recognition functionality.

## Required Models

The application uses the following Vosk models:

### 1. vosk-model-en-us-0.22 (Large Model)
- **Size**: ~70MB
- **Language**: English (US)
- **Accuracy**: High
- **Download**: [https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip](https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip)

### 2. vosk-model-small-en-us-0.15 (Small Model)
- **Size**: ~40MB  
- **Language**: English (US)
- **Accuracy**: Lower, but faster
- **Download**: [https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip](https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip)

## Setup Instructions

1. Download the required models from the links above
2. Extract the zip files into this `Model/` directory
3. Ensure the directory structure looks like:
   ```
   Model/
   ├── README.md (this file)
   ├── vosk-model-en-us-0.22/
   │   ├── am/
   │   ├── conf/
   │   ├── graph/
   │   └── ...
   └── vosk-model-small-en-us-0.15/
       ├── am/
       ├── conf/
       ├── graph/
       └── ...
   ```

## Alternative Models

You can find other Vosk models at [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) for different languages and sizes.

## Note

These model files are excluded from Git due to their large size. Make sure to download them locally for the speech recognition features to work properly.
