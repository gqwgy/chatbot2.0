import config
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import pinecone


class Util:
    def __init__(self):
        super(Util,self).__init__()

        self.ChatOpenAI = ChatOpenAI(

            openai_api_key = config.config['openai']['api_key'],
            model_name = config.config['openai']['chat_model'],
            temperature=0.1 , max_tokens= 2048,
        )
        self.EmbeddingOpenAI = OpenAIEmbeddings(
            openai_api_key =config.config['openai']['api_key'],
            model=config.config['openai']['embedding_model'],
        )

        pinecone.init(
            api_key =config.config['pinecone']['api_key'],
            environment =config.config['pinecone']['environment'],
        )
        #self.VDBPinecone = Pinecone

    @staticmethod
    def concat_chat_message(system_prompt, history, message):
        messages = [SystemMessage(content=system_prompt)]

        for item in history:
            messages.append(HumanMessage(content=item[0]))
            messages.append(AIMessage(content=item[1]))
        messages.append(HumanMessage(content=message))
        return messages
