from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import OpenAIEmbeddings


def process_txt(text):
    text_splitter = CharacterTextSplitter(
        separator= '\n',
        chunk_size= 1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # load model from hugging face
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Sentences are encoded by calling model.encode()
    #embedding = model.encode(chunks)
    #embeddings = OpenAIEmbeddings()
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # similarity search
    end = FAISS.from_texts(chunks, embedding)

    return end



def summarizer(pdf):
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        end = process_txt(text)

        query = "summarize the content of the uploaded pdf file in approximately 5 - 8 lines"

        if query:

            # find embedding that are closest to the query
            docs = end.similarity_search(query)

            # specify model
            OpenAIModel = "gpt-3.5-turbo-16k"
            llm = ChatOpenAI(model=OpenAIModel, temperature=0.8)

            # load question-answering chain with the specified model
            chain = load_qa_chain(llm, chain_type='stuff')

            response =chain.run(input_documents=docs, question=query)

            return response




