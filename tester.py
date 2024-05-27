import os
import re
import string
from typing import List, Tuple, Callable, Dict
from openai import OpenAI
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec
import numpy as np


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

os.environ["PINECONE_API_KEY"] = "5c83f505-ce5d-4790-ad9f-3ab2a148fbd8"
os.environ["OPENAI_API_KEY"] = "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"

class DataSource:
    def __init__(self, data: List[str]):
        """
        Initializes the DataSource object with a list of data.

        Args:
            data (List[str]): The input list of text data.
        """
        self.data = data

    def preprocess_text(self, text: str) -> str:
        """
        Preprocesses the text by lowercasing, removing punctuation, and removing extra whitespace.

        Args:
            text (str): The input text to be preprocessed.

        Returns:
            str: The preprocessed text.
        """
        text = text.lower()  # Convert text to lowercase
        text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
        return text

    def preprocess_text_advanced(self, text: str) -> str:
        """
        Preprocesses the text by lowercasing, removing punctuation, removing extra whitespace,
        removing numbers, removing stop words, and lemmatizing the words.

        Args:
            text (str): The input text to be preprocessed.

        Returns:
            str: The preprocessed and cleaned text.
        """
        # Convert text to lowercase
        text = text.lower() 
        # Remove punctuation and numeric characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize the text
        tokens = nltk.word_tokenize(text)
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        # Join the tokens back into a string
        text = ' '.join(tokens)

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the preprocessed text into a list of words.

        Args:
            text (str): The preprocessed text to be tokenized.

        Returns:
            List[str]: A list of tokens (words) from the text.
        """
        return self.preprocess_text_advanced(text).split()  # Tokenize the preprocessed text by splitting on spaces

    def process_data(self) -> List[str]:
        """
        Processes the data by applying advanced preprocessing to each sentence in the data list.

        Updates the object's processed_data attribute with the cleaned and preprocessed text.

        Args:
            None
        """
        self.processed_data = [self.preprocess_text_advanced(sentence) for sentence in self.data] # Apply preprocessing to each sentence in the data list

        return self.processed_data  # Apply preprocessing to each sentence in the data list

class Embedding:
    def __init__(self, model_name: str, device: str = 'cpu', use_local: bool = False):
        self.model_name = model_name
        self.device = device
        self.use_local = use_local
        # Mapping of model names to their expected dimensions
        self.model_dimensions = {
            'all-MiniLM-L6-v2': 384,  # Example dimension for a SentenceTransformer model
            'jina-v2-base-en-embed': 768,  # Specified dimension for your local model
            # Add other models and their dimensions here
        }
        self.current_model_dimension = self.model_dimensions.get(model_name, None)

        if use_local:
            raise ValueError(f"Not yet implemented")
        else:
            self.model = SentenceTransformer(model_name, device=device)

    def switch_model(self, model_name: str, device: str):
        # Here we assume the model switch is successful and just set the dimension
        if model_name in self.model_dimensions:
            self.current_model_dimension = self.model_dimensions[model_name]
            print(f"Switched to model '{model_name}' with dimension {self.current_model_dimension}")
        else:
            print(f"Model '{model_name}' not recognized. Unable to switch models.")

    def embed(self, text: str) -> List[float]:
        if self.use_local:
          raise ValueError(f"Not yet implemented!")
        else:
            result = self.model.encode(text).tolist()
            if len(result) != self.current_model_dimension:
                print(f"Dimension mismatch detected: Expected {self.current_model_dimension}, got {len(result)}")
            return result

class VectorStorage:
    def __init__(self):
        pass

    def store_vectors(self, vectors: List[List[float]]):
        # Placeholder method for storing vectors
        pass

    def search_vectors(self, query_vector: List[float], top_n: int) -> List[int]:
        # Placeholder method for searching vectors
        pass

class RetrievalAndRanking:
    def __init__(self, data_source: DataSource, embedding: Embedding, vector_storage: VectorStorage):
        self.data_source = data_source
        self.embedding = embedding
        self.vector_storage = vector_storage

    def retrieve_relevant_chunks(self, query: str, top_n: int = 2) -> List[str]:
      """
      Retrieves the most relevant chunks from the data source based on the query.

      Args:
          query (str): The user's query.
          top_n (int): The number of top relevant chunks to retrieve (default is 2).

      Returns:
          List[str]: The list of top relevant chunks from the data source.
      """

      query_tokens = set(self.data_source.tokenize(query))
      similarities: List[Tuple[str, float]] = []

      for chunk in self.data_source.processed_data:
          chunk_tokens = set(self.data_source.tokenize(chunk))
          similarity = len(query_tokens.intersection(chunk_tokens)) / len(
              query_tokens.union(chunk_tokens)
          )
          similarities.append((chunk, similarity))

      similarities.sort(key=lambda x: x[1], reverse=True)
      return [chunk for chunk, _ in similarities[:top_n]]

    def retrieve_relevant_chunks_euclidean(self, query: str, top_n: int = 2) -> List[str]:
      # Create a TF-IDF vectorizer
      vectorizer = TfidfVectorizer()

      # Tokenize each string in the data source and join the tokens back into strings
      tokenized_data_source = [' '.join(self.data_source.tokenize(text)) for text in self.data_source.processed_data]

      # Fit and transform the tokenized data source
      tfidf_matrix = vectorizer.fit_transform(tokenized_data_source)

      # Tokenize the query and join the tokens back into a string
      tokenized_query = ' '.join(self.data_source.tokenize(query))

      # Transform the tokenized query
      query_vector = vectorizer.transform([tokenized_query])

      # Calculate the euclidean distance between the query and each chunk
      similarities = euclidean_distances(query_vector, tfidf_matrix).flatten()

      # Get the indices of the top-n closest chunks
      top_indices = similarities.argsort()[:top_n]

      # Return the top-n closest chunks
      return [self.data_source.processed_data[i] for i in top_indices]

    def retrieve_relevant_chunks_pinecone(self, queries: List[str], top_n: int = 2, filter_metadata: Dict[str, str] = None) -> List[List[str]]:
        relevant_chunks = []

        for query in queries:
            query_embedding = self.embedding.embed(query)
            results = self.vector_storage.search_vectors(query_embedding, top_n)

            filtered_results = []
            if filter_metadata is not None:
                for result in results:
                    metadata = result.get('metadata', {})
                    if all(metadata.get(key) == value for key, value in filter_metadata.items()):
                        filtered_results.append(result)
            else:
                filtered_results = results

            relevant_chunks.append([result['metadata']['text'] for result in filtered_results])

        return relevant_chunks

class UserQuery:
    def __init__(self, query: str):
        self.query = query

class LLM:
    def __init__(self, api_key: str = None, model_name: str = None, device: str = "cuda", consumer_group: str = "mistral"):
        """
        Initializes the LLM object with the specified API key, model name, device, and consumer group.
        Sets up the OpenAI client if an API key is provided.

        Args:
            api_key (str, optional): The API key for accessing the OpenAI service. Default is None.
            model_name (str, optional): The name of the model to be used. Default is None.
            device (str, optional): The device to be used for running the model ("cuda" or "cpu"). Default is "cuda".
            consumer_group (str, optional): The consumer group to be used. Default is "mistral".
        """
        self.api_key = api_key
        self.model_name = model_name
        self.device = device
        self.consumer_group = consumer_group

        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)  # Set up the OpenAI client with the provided API key
        else:
            print('Coming Soon')  # Placeholder message for when API key is not provided

    def switch_model(self, model_name: str, device: str):
        """
        Switches to a different model and device for the LLM. Creates a new reader if necessary.

        Args:
            model_name (str): The name of the new model to switch to.
            device (str): The device to be used for the new model ("cuda" or "cpu").
        """
        current_readers = self.takeoff_client.get_readers()  # Retrieve the current readers from the takeoff client

        # Check if a reader for the desired model already exists
        reader_id = None
        for group, readers in current_readers.items():
            for reader in readers:
                if reader['model_name'] == model_name:  # Check if the desired model is already in use
                    reader_id = reader['reader_id']
                    break
            if reader_id:
                break

        if reader_id:
            print(f"Reader for model '{model_name}' already exists with reader_id: {reader_id}")
        else:
            reader_config = {
                "model_name": model_name,
                "device": device,
                "consumer_group": self.consumer_group
            }

            reader_id, _ = self.takeoff_client.create_reader(reader_config=reader_config)  # Create a new reader
            print(f"Created a new reader with reader_id {reader_id}")

    def answer_query(self, query: str, context: str) -> str:
        """
        Generates an answer to a query based on the provided context using the LLM.

        Args:
            query (str): The user's query.
            context (str): The context to be used for answering the query.

        Returns:
            str: The generated answer to the query.
        """
        prompt = f"Based on the provided context, answer the following query: {query}\n\nContext:\n{context}. Do not use your knowledge, only the context"
        
        if self.api_key:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
                model="gpt-3.5-turbo",  # Specify the model to be used for generating the response
            )
            return chat_completion.choices[0].message.content.strip()  # Return the generated response
        else:
            response = self.takeoff_client.generate(prompt, consumer_group=self.consumer_group)  # Generate response using the takeoff client
            if 'text' in response:
                return response['text'].strip()  # Return the generated response
            else:
                print(f"Error generating response: {response}")
                return "Unable to generate a response."

class PineconeVectorStorage(VectorStorage):
    def __init__(self, index_name: str, embedding: Embedding):
        super().__init__()
        PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
        self.pinecone = Pinecone(api_key=PINECONE_API_KEY)
        if index_name not in self.pinecone.list_indexes().names():
            self.pinecone.create_index(
                name=index_name,
                dimension=embedding.model.get_sentence_embedding_dimension(),
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-west-2')
            )
        self.index = self.pinecone.Index(index_name)

    def store_vectors(self, vectors: List[List[float]], metadatas: List[dict]):
        ids = [str(i) for i in range(len(vectors))]
        records = zip(ids, vectors, metadatas)
        self.index.upsert(vectors=records)

    def search_vectors(self, query_vector: List[float], top_n: int, filter_metadata: Dict[str, str] = None) -> List[dict]:
        query_params = {
            'top_k': top_n,
            'vector': query_vector,
            'include_metadata': True,
            'include_values': False
        }
        if filter_metadata is not None:
            query_params['metadata'] = filter_metadata

        results = self.index.query(**query_params)
        return results['matches']

data = [
    "Carbon trading is a market-based approach to reducing greenhouse gas emissions by providing economic incentives for companies to limit their carbon footprint.",
    "In a carbon trading system, companies are allocated a certain number of carbon credits, which represent the right to emit a specific amount of carbon dioxide or other greenhouse gases.",
    "Companies that emit less than their allocated carbon credits can sell their excess credits to companies that exceed their emissions limits, creating a market for carbon credits.",
    "The goal of carbon trading is to encourage companies to invest in cleaner technologies and adopt more sustainable practices to reduce their emissions and avoid the cost of purchasing additional carbon credits.",
    "Environmental, Social, and Governance (ESG) criteria are a set of standards used by investors to evaluate a company's sustainability and ethical impact.",
    "ESG factors consider a company's environmental impact, such as its carbon footprint, waste management, and use of renewable energy.",
    "Social aspects of ESG include a company's labor practices, diversity and inclusion policies, and community engagement.",
    "Governance factors in ESG assess a company's leadership structure, executive compensation, and transparency in decision-making processes.",
    "Investors are increasingly using ESG criteria to identify companies that are better positioned to manage risks and opportunities related to sustainability and social responsibility.",
    "Companies with strong ESG performance tend to have better long-term financial prospects, as they are more resilient to environmental and social challenges and are favored by environmentally and socially conscious consumers.",
    "Carbon trading and ESG are closely related, as companies with lower carbon emissions and better sustainability practices tend to have higher ESG ratings.",
    "Governments and international organizations are promoting carbon trading and ESG investing as key strategies for mitigating climate change and transitioning to a low-carbon economy.",
]

def process_query(queries: List[str], data_source: DataSource, retrieval_and_ranking: RetrievalAndRanking, llm: LLM, retrieval_method: str = "default") -> List[str]:
    user_queries = [UserQuery(query) for query in queries]
    answers = []

    for user_query in user_queries:
        if retrieval_method == "default":
            relevant_chunks = retrieval_and_ranking.retrieve_relevant_chunks(user_query.query)
            context = "\n".join(relevant_chunks)
            answer = llm.answer_query(user_query.query, context)
        elif retrieval_method == "euclidean":
            relevant_chunks = retrieval_and_ranking.retrieve_relevant_chunks_euclidean(user_query.query)
            context = "\n".join(relevant_chunks)
            answer = llm.answer_query(user_query.query, context)
        elif retrieval_method == "pinecone":
            relevant_chunks = retrieval_and_ranking.retrieve_relevant_chunks_pinecone([user_query.query], filter_metadata={'category': 'finance'})
            answer = "\n".join("\n".join(chunks) for chunks in relevant_chunks)  # Join each list of chunks separately
        else:
            raise ValueError(f"Unknown retrieval method: {retrieval_method}")

        answers.append(answer)

    return answers

def main(data_source:DataSource,
         retrieval_method: str = "default",
         model_choice: str = "openai",
         model_name: str = None,
         device: str = "cpu",
         embedding_model_name = 'all-MiniLM-L6-v2',
         use_local=False,
         index_name="my-index"
         ):
    embedding = Embedding(model_name=embedding_model_name,
                          device=device,
                          use_local=use_local)
    vector_storage = PineconeVectorStorage(index_name, embedding)
    processed_data = data_source.process_data()
    metadatas = [{'text': text, 'category': 'finance'} for text in processed_data]
    vectors = [embedding.embed(text) for text in processed_data]
    vector_storage.store_vectors(vectors, metadatas)

    retrieval_and_ranking = RetrievalAndRanking(data_source, embedding, vector_storage)

    if model_choice == "openai":
        llm = LLM(api_key=os.environ.get("OPENAI_API_KEY"))
    else:
        raise ValueError(f"Not yet implemented")

    while True:
        user_input = input("Enter your query (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        queries = user_input.split(";")  # Split multiple queries separated by semicolon
        answers = process_query(queries, data_source, retrieval_and_ranking, llm, retrieval_method)

        print("User Queries:")
        for query, answer in zip(queries, answers):
            print(f"Query: {query}")
            print(f"Answer: {answer}\n")

data_source = DataSource(data)
data_source.process_data()
main(data_source, retrieval_method='pinecone', index_name="my-index4", embedding_model_name="all-MiniLM-L6-v2")
