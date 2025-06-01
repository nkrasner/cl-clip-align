import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import os
import json
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, ViTModel, ViTImageProcessor, AutoTokenizer
import wandb
import argparse
import random
# import transformers
# from transformers import AutoProcessor, CLIPModel
SENT_MODEL = True
if not SENT_MODEL:
    ROBERTA_MODEL = "xlm-roberta-large"
else:
    ROBERTA_MODEL = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"


class MultiLingualAlignmentDataset(torch.utils.data.Dataset):
    def __init__(self, caption_file, pivot_lang="en"):
        self.tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
        with open(caption_file, "r") as f:
            self.captions = json.load(f)
        
        self.langs = set()
        for caption in self.captions:
            self.langs.update(set(caption["captions"].keys()))
        self.langs = list(self.langs)
        if pivot_lang not in self.langs:
            raise ValueError(f"Pivot language '{pivot_lang}' not found in the caption file.")
        self.langs.remove(pivot_lang)
        
        self.pivot_lang = pivot_lang
        self.pivot_captions = [caption["captions"][pivot_lang] for caption in self.captions]
        for caption in self.captions:
            caption["captions"].pop(pivot_lang)
        # self.max_len = max([max([len(self.tokenizer.encode(caption["captions"][lang])) for caption in self.captions]) for lang in self.langs])
        self.max_len = 128
        


    def __len__(self):
        return len(self.captions)
    
    def preprocess_text(self, text):
        return torch.Tensor(self.tokenizer.encode(text, padding="max_length", max_length=self.max_len)).long()
    
    def __getitem__(self, idx):
        cap_dict = self.captions[idx]
        # caps = [self.preprocess_text(cap_dict["captions"][lang]) for lang in self.langs]
        caps = [self.preprocess_text(cap) for cap in cap_dict["captions"].values()]
        pivot_cap = self.preprocess_text(self.pivot_captions[idx])
        return pivot_cap, *caps
        


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
        
    def set_freeze_encoder(self, val=True):
        for param in self.text_encoder.parameters():
            param.requires_grad = not val
        self.encoder_frozen = val
        

    def mean_pooling(self, model_output, attention_mask=None):
        token_embeddings = model_output.last_hidden_state
        if attention_mask is None:
            attention_mask = torch.ones(token_embeddings.size()[:-1]).to(token_embeddings.device)
        # input_mask_expanded = attention_mask.expand(token_embeddings.size()).float()
        attention_mask = attention_mask.unsqueeze(-1)
        return torch.sum(token_embeddings * attention_mask, 1) / torch.clamp(attention_mask.sum(1), min=1e-9)
    
    def forward(self, pivot, cap, pivot_attn_mask=None, cap_attn_mask=None):
        # Define the forward pass of your model
        # Use the defined layers to process the input and return the output
        if self.pooling == "mean":
            c_enc = self.text_dense(self.mean_pooling(self.text_encoder(cap, attention_mask=cap_attn_mask), attention_mask=cap_attn_mask))
            p_enc = self.text_dense(self.mean_pooling(self.text_encoder(pivot, attention_mask=pivot_attn_mask), attention_mask=pivot_attn_mask))
        elif self.pooling == "cls":
            c_enc = self.text_dense(self.text_encoder(cap, attention_mask=cap_attn_mask).last_hidden_state[:, 0, :])
            p_enc = self.text_dense(self.text_encoder(pivot, attention_mask=pivot_attn_mask).last_hidden_state[:, 0, :])
        else:
            raise ValueError("Invalid pooling type")
        scores = torch.matmul(c_enc, p_enc.T) * self.temp
        return scores




def train(model, device, train_loader, optimizer, epochs, scheduler=None, val_loader=None, val_interval=100, warmup_ratio=None, save_file="models/alignment-model.pt"):
    min_val_loss = float("inf")
    if warmup_ratio is not None:
        model.set_freeze_encoder(True)
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader)
        sum_loss = 0
        pbar.set_description(f"Epoch {epoch}, LR {scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']}")
        for batch_idx, (pivot, *caps) in enumerate(pbar):
            if warmup_ratio is not None:
                if model.encoder_frozen and batch_idx / len(train_loader) >= warmup_ratio:
                    tqdm.write(f"Thawing encoder at step {batch_idx}/{len(train_loader)}")
                    model.set_freeze_encoder(False)
            
            caps = torch.cat(caps, dim=0).to(device)
            cap_attn_mask = torch.ones_like(caps).float().to(device)
            cap_attn_mask[caps == 0] = 0
            pivot = pivot.to(device)
            pivot_attn_mask = torch.ones_like(pivot).float().to(device)
            pivot_attn_mask[pivot == 0] = 0
            optimizer.zero_grad()
            output = model(pivot, caps, pivot_attn_mask, cap_attn_mask)
            loss = F.cross_entropy(output, torch.tile(torch.arange(output.shape[0]//len(train_loader.dataset.langs)), (len(train_loader.dataset.langs),)).to(device))
            loss.backward()
            # Normalize the loss for the batch size and the number of languages so it is comparable across different runs
            sum_loss += loss.item()
            optimizer.step()
            pbar.set_postfix(loss=loss.item(), avg_loss=sum_loss/(batch_idx+1))
            wandb.log({"train_loss": loss.item(), "avg_train_loss": sum_loss/(batch_idx+1), "learning_rate": scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']})
        

            if val_loader is not None and batch_idx > 0 and batch_idx % val_interval == 0:
                model.eval()
                sum_val_loss = 0
                with torch.no_grad():
                    for pivot, *caps in tqdm(val_loader, desc="Validation", leave=False):
                        caps = torch.cat(caps, dim=0).to(device)
                        attn_mask = torch.ones_like(caps).float().to(device)
                        attn_mask[caps == 0] = 0
                        pivot = pivot.to(device)
                        pivot_attn_mask = torch.ones_like(pivot).float().to(device)
                        pivot_attn_mask[pivot == 0] = 0
                        output = model(pivot, caps, pivot_attn_mask, attn_mask)
                        loss = F.cross_entropy(output, torch.tile(torch.arange(output.shape[0]//len(val_loader.dataset.langs)), (len(val_loader.dataset.langs),)).to(device))
                        sum_val_loss += loss.item()
                tqdm.write(f"Epoch: {epoch}, Step: {batch_idx}, Validation Loss: {sum_val_loss/len(val_loader)}")
                wandb.log({"val_loss": sum_val_loss/len(val_loader)})
                if sum_val_loss < min_val_loss:
                    min_val_loss = sum_val_loss
                    torch.save(model.state_dict(), save_file)
            
        if scheduler is not None:
            scheduler.step()











if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--multilingual", action="store_true", help="Use the multilingual dataset")
    argparser.add_argument("--fake_multilingual", action="store_true", help="Use the fake multilingual dataset")
    argparser.add_argument("--multilingual_single", action="store_true", help="Use the multilingual dataset without parallel captions")
    argparser.add_argument("--batch_size", type=int, default=32, help="Set the batch size")
    argparser.add_argument("--learning_rate", type=float, default=1e-5, help="Set the learning rate")
    argparser.add_argument("--warmup_ratio", type=float, default=0.5, help="Set the warmup ratio")
    argparser.add_argument("--total_epochs", type=int, default=10, help="Set the total number of epochs")
    argparser.add_argument("--pivot_lang", type=str, default="en", help="Set the pivot language")
    argparser.add_argument("--pooling", type=str, default="mean", help="Set the pooling type")
    argparser.add_argument("--save_file", type=str, default=None, help="Set the save file")
    
    args = argparser.parse_args()
    
    wandb.init(project="clip-parallel", config=args)
    
    batch_size = args.batch_size  # Set the batch size.
    learning_rate = args.learning_rate  # Set the learning rate.
    warmup_ratio = args.warmup_ratio  # Set the warmup ratio.
    
    model = AlignmentModel(pooling=args.pooling)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if args.multilingual:
        train_dataset = MultiLingualAlignmentDataset(os.path.join("data", "annotations", "captions_train2017_ordered_translated.json"), pivot_lang=args.pivot_lang)
    elif args.fake_multilingual:
        train_dataset = MultiLingualAlignmentDataset(os.path.join("data", "annotations", "captions_train2017_ordered_fakemulti.json"), pivot_lang=args.pivot_lang)
    elif args.multilingual_single:
        train_dataset = MultiLingualAlignmentDataset(os.path.join("data", "annotations", "captions_train2017_ordered_translated_singles.json"), pivot_lang=args.pivot_lang)
    else:
        train_dataset = MultiLingualAlignmentDataset(os.path.join("data", "annotations", "captions_train2017_en_only.json"), pivot_lang=args.pivot_lang)
    print(f"Loaded {len(train_dataset)} training examples")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if args.multilingual:
        val_dataset = MultiLingualAlignmentDataset(os.path.join("data", "annotations", "captions_val2017_ordered_translated.json"), pivot_lang=args.pivot_lang)
    elif args.fake_multilingual:
        val_dataset = MultiLingualAlignmentDataset(os.path.join("data", "annotations", "captions_val2017_ordered_fakemulti.json"), pivot_lang=args.pivot_lang)
    elif args.multilingual_single:
        val_dataset = MultiLingualAlignmentDataset(os.path.join("data", "annotations", "captions_val2017_ordered_translated_singles.json"), pivot_lang=args.pivot_lang)
    else:
        val_dataset = MultiLingualAlignmentDataset(os.path.join("data", "annotations", "captions_val2017_en_only.json"), pivot_lang=args.pivot_lang)
    print(f"Loaded {len(val_dataset)} validation examples")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_epochs = args.total_epochs  # Set the total number of epochs
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch / total_epochs)
    
    # wandb.init(
    #   # set the wandb project where this run will be logged
    #   project="DL Project",
    # #   name=f"{'multilingual' if args.multilingual else 'english'}-clip-{learning_rate}-{total_epochs}-{warmup_ratio}",
    #   # track hyperparameters and run metadata
    #   config={
    #     "lr-init": learning_rate,
    #     "epochs": total_epochs,
    #     "batch_size": batch_size,
    #     "warmup_ratio": warmup_ratio,
    #   }
    # )
    savefile = args.save_file if args.save_file is not None else f"models/{'multi' if args.multilingual else ('fakemulti' if args.fake_multilingual else ('multisingle' if args.multilingual_single else 'en'))}-{args.pivot_lang}pivot-alignment-model-{learning_rate}-{total_epochs}-{warmup_ratio}.pt"
    train(model, device, train_loader, optimizer, epochs=total_epochs, scheduler=scheduler, val_loader=val_loader, val_interval=300, warmup_ratio=warmup_ratio, save_file=savefile)
    wandb.finish()
