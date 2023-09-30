import pinecone

a = pinecone.index('mobot')
documents =  a.query(
            top_k=3,
            include_values=False,
            include_metadata=True,
            vector=question_vector
        )