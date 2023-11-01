# OpenAI Chat Assistant with Context-Aware PDF and Chat History Query

This repository contains a Python script that establishes a conversational assistant capable of answering questions based on a pre-loaded PDF document as well as its own chat history. This makes the conversation flow more naturally and allows for context-aware interactions similar to ChatGPT. The assistant utilizes OpenAI's GPT-4 model for natural language processing, ElevenLabs' API for text-to-speech, and Langchain's ChromaDB for storing vector representations of chat history and PDF content.

## Dependencies

- 'elevenlabs'
- `speech_recognition`
- `wave`
- `langchain`


## Setup

1. Obtain your ElevenLabs API key and input it into the script.
2. Your OpenAI API key should be stored in an environment variable as per best security practices, and is called via LangChain
3. Replace `"path_to_pdf"` with the path to the PDF document you wish to query.
4. Ensure a working microphone is connected to your system.




The program will prompt for voice input, transcribe it, and then produce voice output answering the query.

## Features

- **Context-Aware Conversation**: A separate ChromaDB vector store is created specifically for storing the chat history. This allows the conversation to be context-aware. For example, if you ask for a list of interesting facts and then simply say "any others?", the assistant will understand the context and provide additional facts.

- **Multiple Vector Stores**: Using Langchain's RetrievalQA, the assistant can reference multiple vector stores for different kinds of data. This is particularly useful for tasks that require switching between the PDF content and chat history to provide a relevant response.

## Key Functions

### `record_audio()`

Records audio from a microphone and saves it as a WAV file.

### `transcribe_forever(audio_file_path)`

Transcribes the recorded audio using OpenAI's transcription service.

### `update_agent(chat_history)`

This function updates the conversational agent to include the chat history. The chat history is stored in a separate ChromaDB vector store. This enables the assistant to provide context-aware responses, enhancing the natural flow of the conversation.

## Contributing

If you'd like to contribute, please fork the repository, make your changes, and then submit a pull request. Make sure to include comments in your code to explain your changes.  I am sure there are many ways to improve upon this idea.  There is a noticeable lag between prompts and reponses as LangChain works with OpenAI, but the end result is quite good. 

## License

This project is under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

For more information or queries, please raise an issue or get in touch with the repository owner.
