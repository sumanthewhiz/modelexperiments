from sentence_transformers import SentenceTransformer, util
import torch
import faiss
import numpy

with open("./text/textdata.txt") as file:
    textdata = [line.rstrip() for line in file]
    
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
textdata_embeddings = model.encode(textdata)


vector_dimension = textdata_embeddings.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(textdata_embeddings)
index.add(textdata_embeddings)



query="fruits nutrition"

query_vector = model.encode(query)
_vector = numpy.array([query_vector])
faiss.normalize_L2(_vector)

k = index.ntotal
distances, ann = index.search(_vector, k=k)

#results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})

print("\n")
print(textdata)
print("\n")
print(query)
print("\n")
print(distances[0], ann[0])