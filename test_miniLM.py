from sentence_transformers import SentenceTransformer, util
import torch

with open("./text/textdata.txt") as file:
    textdata = [line.rstrip() for line in file]
    
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
textdata_embeddings = model.encode(textdata)

#print(embeddings)

query="fruits nutrition"

print("\nQuery:", query)

query_embedding = model.encode(query)

# Find the closest 3 sentences of the text corpus for each query sentence based on cosine similarity
top_k = min(3, len(textdata))

cos_scores = util.cos_sim(query_embedding, textdata_embeddings)[0]
top_results = torch.topk(cos_scores, k=top_k)

print("\nTop 3 most similar sentences in text corpus:")

for score, idx in zip(top_results[0], top_results[1]):
    print(idx.item()+1, ":", textdata[idx], "(Score: {:.4f})".format(score))

