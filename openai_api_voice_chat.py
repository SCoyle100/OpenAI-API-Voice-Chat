from langchain.chat_models.openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.callbacks import get_openai_callback
import speech_recognition as sr
import openai
import wave

from elevenlabs import set_api_key, generate, play, stream
set_api_key("enter_here")


def speak(text):
    audio = generate(text=text, voice="Dorothy", model="eleven_multilingual_v1")
    stream(audio)

whileWaiting = "Ok! Give me a few seconds to look into that."
whileWaiting2 = "Sorry, it's taking just a little longer than expected"


def record_audio():
    # Load the speech recognizer and set the initial energy threshold and pause threshold
    r = sr.Recognizer()
    r.energy_threshold = 300
    r.pause_threshold = 0.8
    r.dynamic_energy_threshold = False

    with sr.Microphone(sample_rate=16000) as source:
        print("Say something!")
        # Get and save audio to wav file
        audio = r.listen(source)
        
        # Define the file name
        filename = "recording.wav"
        
        # Save the audio data to a file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(audio.sample_width)
            wf.setframerate(16000)
            wf.writeframes(audio.get_wav_data())
        
        # Return the file name
        return filename
        
    

def transcribe_forever(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        result = openai.Audio.transcribe("whisper-1", audio_file)
    predicted_text = result["text"]
    return predicted_text




# Create instance of OpenAI LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.1, verbose=True)
# Create and load PDF Loader
loader = PyPDFLoader("path_to_pdf")
# Split loaded pdf
pages = loader.load_and_split()

# Load documents into vector database aka ChromaDB
embeddings = OpenAIEmbeddings()
pdf_store = Chroma.from_documents(pages, embeddings, collection_name="pdf_name")
pdf_retrievalQA = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=pdf_store.as_retriever()
)

# Initialize an empty chat history
chat_history = ""

# Initialize tools with only PDF vector store initially
tools = [
    Tool(
        name="PDF QA System",
        func=pdf_retrievalQA.run,
        description="useful for when you need to answer questions about the PDF. Input should be a fully formed question.",
    ),
]

agent = initialize_agent(
    tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True
)

def split_text_into_chunks(text, chunk_size):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

class DocumentWrapper:
    def __init__(self, text):
        self.page_content = text
        self.metadata = None  # or {} if you prefer an empty dictionary

# Define a predefined prompt
PREDEFINED_PROMPT = "Here is the previous conversation for context:\n"


def update_agent(chat_history):
    chat_texts = split_text_into_chunks(chat_history, 1000)  # Split chat history into chunks of 1000 characters
    # Wrap each text chunk in a DocumentWrapper object
    wrapped_texts = [DocumentWrapper(text) for text in chat_texts]
    chat_store = Chroma.from_documents(wrapped_texts, embeddings, collection_name="ChatHistory")
    chat_retrievalQA = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=chat_store.as_retriever()
    )
    # Update tools to include both PDF and chat history vector stores
    tools.extend([
        Tool(
            name="Chat History QA System",
            func=chat_retrievalQA.run,
            description="useful for when you need to answer questions about the chat history. Input should be a fully formed question.",
        ),
    ])
    return initialize_agent(
        tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=False, handle_parsing_errors=True
    )


first_prompt = True  # Initialize a flag to check if it's the first prompt

while True:
    prompt = transcribe_forever(record_audio())
    if prompt:
          # Convert text to audio
        audio_data = generate(
            text=whileWaiting,
            voice="Dorothy",  # Or any other voice
            model="eleven_multilingual_v1"  # Or any other model
        )
        # Stream the audio data
        play(audio_data)

        if first_prompt:
            # If it's the first prompt, do not prepend the predefined prompt or chat history
            response = agent.run(prompt)
            first_prompt = False  # Update the flag as the first prompt has been handled
        else:
            # Prepend predefined prompt and chat history to user's input for subsequent prompts
            prompt_with_context = PREDEFINED_PROMPT + chat_history + prompt
            response = agent.run(prompt_with_context)
        
        print(response)
        #engine.say(response)
        #engine.runAndWait()
        audio_data = generate(
            text=response,
            voice="Dorothy",  # Or any other voice
            model="eleven_multilingual_v1"  # Or any other model
        )
        # Stream the audio data
        play(audio_data)
        # Update chat history
        chat_history += f"User: {prompt}\nAgent: {response}\n"
        
        # If this is after the first prompt, update the agent to include chat history vector store
        if len(chat_history.split('\n')) > 2:
            agent = update_agent(chat_history)