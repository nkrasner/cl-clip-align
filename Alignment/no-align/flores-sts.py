from datasets import load_dataset
import open_clip
import tqdm
import torch
import numpy as np
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

BATCH_SIZE = 16

# Load the dataset
dataset = load_dataset("facebook/flores", "all", trust_remote_code=True, split="devtest")


ROBERTA_MODEL = "xlm-roberta-large"
# ROBERTA_MODEL = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
# ROBERTA_MODEL = "sentence-transformers/LaBSE"

# distance = "cosine"
distance = "xsim"

# # Load the clip model
# model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(clip_model_name, cache_dir="/projects/antonis/nate/universal-encoding/clip-bitext")
# tokenizer = AutoTokenizer.from_pretrained(clip_model_name, cache_dir="/projects/antonis/nate/universal-encoding/clip-bitext")

# model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
# tokenizer = open_clip.get_tokenizer('ViT-B-32')

def mean_pooling(model_output, attention_mask=None):
        token_embeddings = model_output.last_hidden_state
        if attention_mask is None:
            attention_mask = torch.ones(token_embeddings.size()[:-1]).to(token_embeddings.device)
        # input_mask_expanded = attention_mask.expand(token_embeddings.size()).float()
        attention_mask = attention_mask.unsqueeze(-1)
        return torch.sum(token_embeddings * attention_mask, 1) / torch.clamp(attention_mask.sum(1), min=1e-9)

def encode_mean_pool(texts, tokenizer, encoder):
    tokenizer.src_lang = lang
    with torch.inference_mode():
        batch = []
        for text in texts:
            batch.append(torch.Tensor(tokenizer.encode(text, padding="max_length", max_length=500)).long().to(device))
        batch = torch.stack(batch)
        attn_mask = batch != 0
        seq_embs = encoder(batch, attention_mask=attn_mask)
        mean_emb = mean_pooling(seq_embs, attn_mask)
    return mean_emb

def xsim(X, Y, k=4):
    # Calculate the cosine distance between each pair of vectors in X and Y
    cos_xy = 1 - ((X @ Y.T) / (np.linalg.norm(X, axis=1)[:, None] * np.linalg.norm(Y, axis=1)[None, :]))
    
    # Calculate the cosine distance between each pair of vectors within X and Y
    cos_xx = 1 - ((X @ X.T) / (np.linalg.norm(X, axis=1)[:, None] * np.linalg.norm(X, axis=1)[None, :]))
    
    cos_yy = 1 - ((Y @ Y.T) / (np.linalg.norm(Y, axis=1)[:, None] * np.linalg.norm(Y, axis=1)[None, :]))
    
    # Calculate the sum of cosine distances to the k nearest neighbors for each vector in X and Y
    cos_xz = np.sum(np.sort(cos_xx, axis=1)[:, 1:(k+1)], axis=1)/(2*k)
    cos_yz = np.sum(np.sort(cos_yy, axis=1)[:, 1:(k+1)], axis=1)/(2*k)
    
    denom = np.add.outer(cos_xz, cos_yz)
    
    # Compute the final metric
    return cos_xy / denom

model = AutoModel.from_pretrained(ROBERTA_MODEL)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
langs = [lang.replace("sentence_", "") for lang in dataset.features.keys() if "sentence_" in lang]

if not os.path.exists(f"vectors/flores-{ROBERTA_MODEL}"):
    os.makedirs(f"vectors/flores-{ROBERTA_MODEL}")
    
# Produce or retrieve the vectors
vectors = {}
for lang in langs:
    if not os.path.exists(f"vectors/flores-{ROBERTA_MODEL}/{lang}-vecs.tsv"):
        
        loader = torch.utils.data.DataLoader(dataset[f"sentence_{lang}"], batch_size=BATCH_SIZE, shuffle=False)
        vectors[lang] = []
        for texts in tqdm.tqdm(loader, desc=f"Processing {lang}"):
            tokens = []
            for text in texts:
                tokens.append(torch.Tensor(tokenizer.encode(text, padding="max_length", max_length=500)).long().to(device))
            tokens = torch.stack(tokens)
            attn_mask = torch.stack([(torch.ones_like(t) * (t != 0).float()).float().to(device) for t in tokens])
            # Average over the tokens
            vectors[lang] += encode_mean_pool(texts, tokenizer, model).cpu().detach().numpy().tolist()
        
        with open(f"vectors/flores-{ROBERTA_MODEL}/{lang}-vecs.tsv", "w") as f:
            for vec in vectors[lang]:
                f.write("\t".join(map(str, vec)) + "\n")
                
        vectors[lang] = np.array(vectors[lang])
        
        
        # with open(f"vectors/flores-{lang}-texts.tsv", "w") as f:
        #     for text in dataset[f"sentence_{lang}"]:
        #         f.write(text + "\n")
    
    else:
        with open(f"vectors/flores-{ROBERTA_MODEL}/{lang}-vecs.tsv", "r") as f:
            vectors[lang] = np.array([list(map(float, line.strip().split("\t"))) for line in f])


# Calculate the similarities between the vectors in english and the vectors in the other languages
# Then check the accuracy of the nearest neighbor along the diagonal
similarities = {}
eng_x_accuracies = {}
x_eng_accuracies = {}

for lang in langs:
    if lang == "eng_Latn":
        continue
    print(f"Calculating similarities between eng_Latn and {lang}...")
    if distance == "cosine":
        similarities[lang] = cosine_similarity(vectors["eng_Latn"], vectors[lang])
    elif distance == "xsim":
        similarities[lang] = 1 - xsim(vectors["eng_Latn"], vectors[lang])
    else:
        raise ValueError("Invalid distance metric")
    
    eng_x_accuracy = np.sum(np.argmax(similarities[lang], axis=1) == np.arange(len(vectors["eng_Latn"]))) / len(vectors["eng_Latn"])
    x_eng_accuracy = np.sum(np.argmax(similarities[lang], axis=0) == np.arange(len(vectors[lang]))) / len(vectors[lang])
    
    print(f"Accuracy eng_Latn -> {lang}: {eng_x_accuracy}")
    print(f"Accuracy {lang} -> eng_Latn: {x_eng_accuracy}")
    
    eng_x_accuracies[lang] = eng_x_accuracy
    x_eng_accuracies[lang] = x_eng_accuracy


# Save the results
with open(f"flores-{ROBERTA_MODEL.split('/')[-1]}-{distance}_accuracies{'-sentmodel'}.json", "w") as f:
    json.dump({"eng_Latn->x": eng_x_accuracies, "x->eng_Latn": x_eng_accuracies}, f)