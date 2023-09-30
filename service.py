import util
import prompt
import pinecone


class Service:
    def __init__(self):
        super(Service, self).__init__()
        self.util = util.Util()

    def retrival_inference_answer(self, message, history):

        question_vector = self.util.EmbeddingOpenAI.embed_query(message)

        # index = self.util.VDBPinecone.get_pinecone_index('mobot')
        index = pinecone.Index('mobot')
        documents = index.query(
            top_k=3,
            include_values=False,
            include_metadata=True,
            vector=question_vector
        )

        retrieval = ''
        if len(documents['matches']) == 0:
            retrieval = '没有找到相关数据'

        for doc in documents['matches'] :
            if float(doc['score']) > 0.75:
                retrieval += f"问题:{doc['metadata']['question']} 答案:{doc['metadata']['answer']}"

        system_prompt = prompt.GENERIC_SYSTEM_PROMPT

        user_prompt = f'历史对话:\n{history}\n\n知识库:\n\n{retrieval}\n\n用户问题：\n\n{message}'

        messages = self.util.concat_chat_message(system_prompt, [], user_prompt)

        response = self.util.ChatOpenAI(messages)

        return response.content
