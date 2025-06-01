import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import transformers
from transformers import AutoTokenizer, AutoModel, ViTModel
from tqdm import tqdm
import json
import sys
import argparse
import numpy as np
import gc

ROBERTA_MODEL = "xlm-roberta-large"
VIT_MODEL = "google/vit-base-patch16-224-in21k"




class ClipModel(nn.Module):
    def __init__(self, pooling="mean", baseline=False):
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
        self.baseline = baseline
        
    def encode_text(self, cap, attn_mask=None):
        if self.pooling == "mean":
            if attn_mask is None:
            # if no mask is provided, assume all tokens are valid
                attn_mask = torch.ones_like(cap).float()
            mask = attn_mask
            seq_embs = self.text_encoder(cap, attention_mask=attn_mask).last_hidden_state
            t_enc = (seq_embs * mask.unsqueeze(-1)).sum(1) / mask.unsqueeze(-1).sum(1)
            if not self.baseline:
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

# Model for pivot aligned
class AlignmentModel(nn.Module):
    def __init__(self, pooling="mean"):
        super(AlignmentModel, self).__init__()
        # Define your layers here
        # Encoders
        self.text_encoder = AutoModel.from_pretrained(ROBERTA_MODEL)
        self.text_dense = nn.Linear(self.text_encoder.config.hidden_size, 512)
        self.temp = nn.Parameter(torch.ones(1))
        self.pooling = pooling
        self.encoder_frozen = False
    
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

class NLI(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_hidden=2):
        super(NLI, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden = num_hidden
        self.input = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList()
        for i in range(num_hidden):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.output = nn.Linear(hidden_size, 3)
        
    def forward(self, x):
        x = F.relu(self.input(x))
        for i in range(self.num_hidden):
            x = F.relu(self.layers[i](x))
        x = self.output(x)
        x = F.softmax(x, dim=1)
        return x
    
def preprocess_vecs(p_vecs, h_vecs):
    return torch.cat([p_vecs, h_vecs, torch.abs(p_vecs - h_vecs), p_vecs * h_vecs], dim=-1)

# Text encoder model
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
    if "pivot" in model_name:
        model = AlignmentModel()
    else:
        model = ClipModel(baseline=model_name == "baseline")
    if model_name != "baseline":
        model.load_state_dict(torch.load(os.path.join("..", "models", model_name)))
    return model, tokenizer

class TupleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
 
def generate_vecs(model_name, vec_name):
    # Generate the vectors
    model, tokenizer = load_model(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data_path = os.path.join("data", vec_name+".json")
    with open(data_path) as f:
        data = json.load(f)
    data = TupleDataset(data)
    data = torch.utils.data.DataLoader(data, batch_size=128, shuffle=False)
    gen_pbar = tqdm(data, desc="Generating vectors for " + vec_name)
    
    vecs = torch.tensor([])
    model.eval()
    vecs_list = []
    for i, (sent1, sent2, _) in enumerate(gen_pbar):
        tok1 = tokenizer.batch_encode_plus(sent1, return_tensors="pt", padding=True).input_ids.long()
        tok2 = tokenizer.batch_encode_plus(sent2, return_tensors="pt", padding=True).input_ids.long()
        
        tok1 = tok1.to(device)
        tok2 = tok2.to(device)
        attn_mask1 = (tok1 != 0).float()
        attn_mask2 = (tok2 != 0).float()
        
        with torch.no_grad():
            # vecs.append(preprocess_vecs(model.encode_text(tok1),
            #                             model.encode_text(tok2)).cpu().tolist())
            new_vecs = preprocess_vecs(model.encode_text(tok1, attn_mask1),
                                        model.encode_text(tok2, attn_mask2)).detach().cpu()
            vecs_list.append(new_vecs)
            
        if i % 100 == 0:
            # Clean up memory every 100 iterations

            torch.cuda.empty_cache()
            gc.collect()
    # Concatenate the vecs list
    vecs = torch.cat(vecs_list, dim=0)
    # Save the vecs tensor
    torch.save(vecs, os.path.join("vectors", model_name, vec_name))
        
        
    
        
def load_vecs(model_name, vec_name):
    vec_path = os.path.join("vectors", model_name, vec_name)
    if not os.path.exists(vec_path):
        os.makedirs(os.path.join("vectors", model_name), exist_ok=True)
        generate_vecs(model_name, vec_name)
    
    vecs = torch.load(vec_path)
    print("Loaded vectors from", vec_path)
    
    label_path = os.path.join("data", vec_name+".json")
    with open(label_path) as f:
        samples = json.load(f)
        labels = [sample[2] for sample in samples]
    
    label_ids = {"entailment": 0, "neutral": 1, "contradiction": 2}
    for sample in samples:
        if sample[2] not in label_ids:
            print(sample)
    labels = torch.tensor([label_ids[label] for label in labels])
    return vecs, labels

def train(model_name, **config):
    train_vecs, train_labels = load_vecs(model_name, "train")
    dev_vecs, dev_labels = load_vecs(model_name, "dev")
    model = NLI(config["input_size"], config.get("hidden_size", 512), config.get("num_hidden", 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_vecs = train_vecs
    train_labels = train_labels
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_vecs, train_labels), batch_size=config.get("batch_size", 32), shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(config.get("epochs", 10)):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        sum_loss = 0
        sum_acc = 0
        for i, (vecs, labels) in enumerate(train_pbar):
            vecs = vecs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(vecs)
            loss = criterion(output, labels)
            acc = (output.argmax(1) == labels).float().mean()
            sum_loss += loss.item()
            sum_acc += acc.item()
            loss.backward()
            optimizer.step()
            train_pbar.set_postfix({"loss": sum_loss / (i+1), "acc": sum_acc / (i+1)})
        model.eval()
        with torch.no_grad():
            dev_vecs = dev_vecs.to(device)
            dev_labels = dev_labels.to(device)
            dev_output = model(dev_vecs)
            dev_loss = criterion(dev_output, dev_labels)
            dev_acc = (dev_output.argmax(1) == dev_labels).float().mean()
            print(f"Epoch {epoch}: Dev loss {dev_loss}, Dev acc {dev_acc}")
            if dev_acc > best_acc:
                best_acc = dev_acc
                torch.save(model.state_dict(), os.path.join("models", model_name))
                print("Model saved")

def test(model_name, lang, **config):
    test_vecs, test_labels = load_vecs(model_name, f"{lang}_test")
    model = NLI(config["input_size"], config.get("hidden_size", 512), config.get("num_hidden", 2))
    model.load_state_dict(torch.load(os.path.join("models", model_name)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_vecs = test_vecs.to(device)
    test_labels = test_labels.to(device)
    model.eval()
    with torch.no_grad():
        test_output = model(test_vecs)
        acc = (test_output.argmax(1) == test_labels).float().mean()
        print(f"Test acc for {lang}: {acc}")

def main():
    argparser = argparse.ArgumentParser()
    # Model name is positional
    argparser.add_argument("model_name", type=str)
    argparser.add_argument("--input_size", type=int, default=512*4)
    argparser.add_argument("--hidden_size", type=int, default=512)
    argparser.add_argument("--num_hidden", type=int, default=2)
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--epochs", type=int, default=10)
    argparser.add_argument("--lr", type=float, default=1e-3)
    
    args = argparser.parse_args()
    model_name = args.model_name
    config = vars(args)
    config.pop("model_name")
    if not os.path.exists(os.path.join("models", model_name)):
        train(model_name, **config)

    # langs = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
    langs = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh", "aym", "bzd", "cni", "gn", "hch", "nah", "oto", "quy", "shp", "tar"]
    
    print("Testing model:", model_name)
    for lang in langs:
        test(model_name, lang, **config)
    
    print("Done!")

if __name__ == "__main__":
    main()