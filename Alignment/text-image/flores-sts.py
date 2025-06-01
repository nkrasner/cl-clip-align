from datasets import load_dataset
import tqdm
import torch
import numpy as np
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, ViTModel, ViTImageProcessor, AutoTokenizer
import torch.nn as nn
import sys

BATCH_SIZE = 8

# Load the dataset
dataset = load_dataset("facebook/flores", "all", trust_remote_code=True, split="devtest")

ROBERTA_MODEL = "xlm-roberta-large"
# ROBERTA_MODEL = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
# ROBERTA_MODEL = "sentence-transformers/LaBSE"
VIT_MODEL = "google/vit-base-patch16-224-in21k"
POOLING = "mean"
distance = "cosine"
# distance = "xsim"

PIVOT_LANG = "eng_Latn"
# PIVOT_LANG = "quy_Latn"

# # Load the clip model
# model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(clip_model_name, cache_dir="/projects/antonis/nate/universal-encoding/clip-bitext")
# tokenizer = AutoTokenizer.from_pretrained(clip_model_name, cache_dir="/projects/antonis/nate/universal-encoding/clip-bitext")

# model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
# tokenizer = open_clip.get_tokenizer('ViT-B-32')

MODEL_FILE = sys.argv[1]

class ClipModel(nn.Module):
    def __init__(self, pooling="mean"):
        super(ClipModel, self).__init__()
        # Define your layers here
        # Encoders
        self.text_encoder = AutoModel.from_pretrained(ROBERTA_MODEL)
        self.image_encoder = ViTModel.from_pretrained(VIT_MODEL)
        self.text_dense = nn.Linear(self.text_encoder.config.hidden_size, 512)
        self.image_dense = nn.Linear(self.image_encoder.config.hidden_size, 512)
        self.temp = nn.Parameter(torch.ones(1))
        self.pooling = pooling
        self.encoders_frozen = False
        
    def encode_text(self, cap, attn_mask=None):
        if self.pooling == "mean":
            if attn_mask is None:
            # if no mask is provided, assume all tokens are valid
                attn_mask = torch.ones_like(cap).float()
            mask = attn_mask
            seq_embs = self.text_encoder(cap, attention_mask=attn_mask).last_hidden_state
            t_enc = (seq_embs * mask.unsqueeze(-1)).sum(1) / mask.unsqueeze(-1).sum(1)
            t_enc = self.text_dense(t_enc)
            return t_enc.detach()
        elif self.pooling == "cls":
            return self.text_dense(self.text_encoder(cap, attention_mask=attn_mask).last_hidden_state[:, 0, :]).detach()
        else:
            raise ValueError("Invalid pooling method")
    
    def encode_image(self, img):
        if img is None:
            return torch.zeros(512)
        
        return self.image_dense(self.image_encoder(pixel_values=img.pixel_values).last_hidden_state[:, 0, :]).detach()

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

model = ClipModel(pooling=POOLING)
model.load_state_dict(torch.load(MODEL_FILE))

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
langs = [lang.replace("sentence_", "") for lang in dataset.features.keys() if "sentence_" in lang]

if not os.path.exists(f"vectors/flores-{MODEL_FILE.split('/')[-1]}"):
    os.makedirs(f"vectors/flores-{MODEL_FILE.split('/')[-1]}")
    
# Produce or retrieve the vectors
vectors = {}
for lang in langs:
    if not os.path.exists(f"vectors/flores-{MODEL_FILE.split('/')[-1]}/{lang}-vecs.tsv"):
        
        loader = torch.utils.data.DataLoader(dataset[f"sentence_{lang}"], batch_size=BATCH_SIZE, shuffle=False)
        vectors[lang] = []
        for texts in tqdm.tqdm(loader, desc=f"Processing {lang}"):
            tokens = []
            for text in texts:
                tokens.append(torch.Tensor(tokenizer.encode(text, padding="max_length", max_length=500)).long().to(device))
            tokens = torch.stack(tokens)
            # attn_mask = torch.stack([(torch.ones_like(t) * (t != 0).float()).float().to(device) for t in tokens])
            attn_mask = (tokens != 0).float()
            
            vectors[lang] += model.encode_text(tokens, attn_mask).cpu().numpy().tolist()
        
        with open(f"vectors/flores-{MODEL_FILE.split('/')[-1]}/{lang}-vecs.tsv", "w") as f:
            for vec in vectors[lang]:
                f.write("\t".join(map(str, vec)) + "\n")
                
        vectors[lang] = np.array(vectors[lang])
        
        
        # with open(f"vectors/flores-{lang}-texts.tsv", "w") as f:
        #     for text in dataset[f"sentence_{lang}"]:
        #         f.write(text + "\n")
    
    else:
        with open(f"vectors/flores-{MODEL_FILE.split('/')[-1]}/{lang}-vecs.tsv", "r") as f:
            vectors[lang] = np.array([list(map(float, line.strip().split("\t"))) for line in f])


# Calculate the similarities between the vectors in english and the vectors in the other languages
# Then check the accuracy of the nearest neighbor along the diagonal
similarities = {}
eng_x_accuracies = {}
x_eng_accuracies = {}

for lang in langs:
    if lang == PIVOT_LANG:
        continue
    print(f"Calculating similarities between {PIVOT_LANG} and {lang}...")
    if distance == "cosine":
        similarities[lang] = cosine_similarity(vectors[PIVOT_LANG], vectors[lang])
    elif distance == "xsim":
        similarities[lang] = 1 - xsim(vectors[PIVOT_LANG], vectors[lang])
    else:
        raise ValueError("Invalid distance metric")
    
    eng_x_accuracy = np.sum(np.argmax(similarities[lang], axis=1) == np.arange(len(vectors[PIVOT_LANG]))) / len(vectors[PIVOT_LANG])
    x_eng_accuracy = np.sum(np.argmax(similarities[lang], axis=0) == np.arange(len(vectors[lang]))) / len(vectors[lang])
    
    print(f"Accuracy {PIVOT_LANG} -> {lang}: {eng_x_accuracy}")
    print(f"Accuracy {lang} -> {PIVOT_LANG}: {x_eng_accuracy}")
    
    eng_x_accuracies[lang] = eng_x_accuracy
    x_eng_accuracies[lang] = x_eng_accuracy


# Save the results
with open(f"flores-{MODEL_FILE.split('/')[-1]}-{distance}_{PIVOT_LANG}_accuracies.json", "w") as f:
    json.dump({f"{PIVOT_LANG}->x": eng_x_accuracies, f"x->{PIVOT_LANG}": x_eng_accuracies}, f)
