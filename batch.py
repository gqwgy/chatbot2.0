import util
import csv
import pinecone

file_path = './data/knowledge_base.csv'

with open(file_path, 'r',encoding ='utf-8') as file:
    reader = csv.reader(file)

    questions, answers, metadata = [], [], []
    for row in reader:
        questions.append(row[0])
        answers.append(row[1])
        metadata.append({'question': row[0],'answer':row[1]})
    print(f'CSV文件读取到了{len(metadata)}行数据')

    u = util.Util()
    vectors = u.EmbeddingOpenAI.embed_documents(questions)
    ids = list(range(1,len(questions)+1))

    data_list = []
    for i in range(len(questions)):
        data_list.append([ids[i],questions[i],answers[i],vectors[i]])

    save_file_path = './data/knowledge_base_with_embed.csv'

    with open(save_file_path,'w',newline='') as f:
        writer = csv.writer(f)

    print(f'写入了{len(vectors)}条Document在{save_file_path}文件中')

    documents = []
    for i in range(len(vectors)):
        documents.append(
            (
                str(ids[i]),
                vectors[i],
                metadata[i]
            )

        )
    #index = u.VDBPinecone.get_pinecone_index('mobot')
    index = pinecone.Index('mobot')
    response = index.upsert(
        vectors = documents

    )
    print(f"Pinecone数据库写入了{response['upserted_count']}条Document")