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
# import sentence_transformers
import wandb
import argparse
import random
# import transformers
# from transformers import AutoProcessor, CLIPModel

ROBERTA_MODEL = "xlm-roberta-large"
# ROBERTA_MODEL = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
# ROBERTA_MODEL = "sentence-transformers/LaBSE"
VIT_MODEL = "google/vit-base-patch16-224-in21k"


class MultiLingualCLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, caption_file, pivot_file=None, scramble_images=False):
        self.images = {}
        for image_file in os.listdir(image_dir):
            index = int(image_file.split(".")[0])
            # with Image.open(os.path.join(image_dir, image_file)) as img:
            #     self.images[index] = img.copy()
            self.images[index] = os.path.join(image_dir, image_file)
        # Shuffle which image corresponds to which captions
        # This way we can see whether the images are actually important for aligning the captions
        if scramble_images:
            scrambled_indices = list(self.images.keys())
            random.shuffle(scrambled_indices)
            self.images = {k: self.images[v] for k, v in zip(self.images.keys(), scrambled_indices)}
            
        self.tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
        self.image_processor = ViTImageProcessor.from_pretrained(VIT_MODEL)
        with open(caption_file, "r") as f:
            self.captions = json.load(f)
        if pivot_file is not None:
            with open(pivot_file, "r") as f:
                self.pivot_captions = json.load(f)
            pivot_lang = list(self.pivot_captions[0]["captions"].keys())[0]
            # Remove captions that are from the pivot language
            for i in range(len(self.captions)):
                if pivot_lang in self.captions[i]["captions"]:
                    self.captions[i]["captions"].pop(pivot_lang)
            temp_captions = []
            temp_images = {}
            temp_pivots = []
            for i in range(len(self.captions)):
                if self.captions[i]["captions"] != {}:  # If there are no more captions left, remove the example
                    temp_captions.append(self.captions[i])
                    temp_images[self.captions[i]["image_id"]] = self.images[self.captions[i]["image_id"]]
                    temp_pivots.append(self.pivot_captions[i])
            self.captions = temp_captions
            self.images = temp_images
            self.pivot_captions = temp_pivots
            self.langs = [pivot_lang] + list(self.captions[0]["captions"].keys())
        else:
            self.pivot_captions = None
            # Dont use self.langs for anything since it is not accurate unless parallel captions are used
            self.langs = list(self.captions[0]["captions"].keys())
            
        # self.max_len = max([max([len(self.tokenizer.encode(caption["captions"][lang])) for caption in self.captions]) for lang in self.langs])
        self.max_len = 128
        


    def __len__(self):
        return len(self.captions)
    
    def preprocess_text(self, text):
        return torch.Tensor(self.tokenizer.encode(text, padding="max_length", max_length=self.max_len)).long()
    
    def __getitem__(self, idx):
        cap_dict = self.captions[idx]
        # caps = [self.preprocess_text(cap_dict["captions"][lang]) for lang in self.langs]
        if self.pivot_captions is not None:
            pivot = list(self.pivot_captions[idx]["captions"].values())[0]
            caps = [self.preprocess_text(pivot)] + [self.preprocess_text(cap) for cap in cap_dict["captions"].values()]
        else:
            caps = [self.preprocess_text(cap) for cap in cap_dict["captions"].values()]

        
        with Image.open(self.images[cap_dict["image_id"]]) as img:
            image = self.image_processor.preprocess(img.convert("RGB"), return_tensors="pt")
        # image = self.image_processor(self.images[cap_dict["image_id"]], return_tensors="pt")
        return image, *caps
        


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
        self.image_frozen = False
        self.text_frozen = False
        
    def set_freeze_encoders(self, val=True, encoders="both"):
        if encoders == "both" or encoders == "text":
            for param in self.text_encoder.parameters():
                param.requires_grad = not val
            self.text_frozen = val
        if encoders == "both" or encoders == "image":
            for param in self.image_encoder.parameters():
                param.requires_grad = not val
            self.image_frozen = val
        self.encoders_frozen = self.text_frozen and self.image_frozen
    
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask=None):
        token_embeddings = model_output.last_hidden_state
        if attention_mask is None:
            attention_mask = torch.ones(token_embeddings.size()[:-1]).to(token_embeddings.device)
        # input_mask_expanded = attention_mask.expand(token_embeddings.size()).float()
        attention_mask = attention_mask.unsqueeze(-1)
        return torch.sum(token_embeddings * attention_mask, 1) / torch.clamp(attention_mask.sum(1), min=1e-9)

    def forward(self, pivot, cap, attn_mask=None, pivot_attn_mask = None, img_pivot=True):
        # Define the forward pass of your model
        # Use the defined layers to process the input and return the output
        if img_pivot:
            if self.pooling == "mean":
                t_enc = self.text_dense(self.mean_pooling(self.text_encoder(cap, attention_mask=attn_mask), attention_mask=attn_mask))
                p_enc = self.image_dense(self.mean_pooling(self.image_encoder(pixel_values=pivot.pixel_values)))
                # t_enc = self.text_dense(self.text_encoder(cap, attention_mask=attn_mask).last_hidden_state.mean(dim=1))
                # p_enc = self.image_dense(self.image_encoder(pixel_values=pivot.pixel_values).last_hidden_state.mean(dim=1))
            elif self.pooling == "cls":
                t_enc = self.text_dense(self.text_encoder(cap, attention_mask=attn_mask).last_hidden_state[:, 0, :])
                p_enc = self.image_dense(self.image_encoder(pixel_values=pivot.pixel_values).last_hidden_state[:, 0, :])
            else:
                raise ValueError("Invalid pooling type")
        else:
            if self.pooling == "mean":
                t_enc = self.text_dense(self.mean_pooling(self.text_encoder(cap, attention_mask=attn_mask), attention_mask=attn_mask))
                p_enc = self.text_dense(self.mean_pooling(self.text_encoder(pivot, attention_mask=pivot_attn_mask), attention_mask=pivot_attn_mask))
                # t_enc = self.text_dense(self.text_encoder(cap, attention_mask=attn_mask).last_hidden_state.mean(dim=1))
                # p_enc = self.text_dense(self.text_encoder(pivot, attention_mask=pivot_attn_mask).last_hidden_state.mean(dim=1))
            elif self.pooling == "cls":
                t_enc = self.text_dense(self.text_encoder(cap, attention_mask=attn_mask).last_hidden_state[:, 0, :])
                p_enc = self.text_dense(self.text_encoder(pivot, attention_mask=pivot_attn_mask).last_hidden_state[:, 0, :])
            else:
                raise ValueError("Invalid pooling type")
        scores = torch.matmul(t_enc, p_enc.T) * self.temp
        return scores




def train(model, device, train_loader, optimizer, epochs, scheduler=None, val_loader=None, val_interval=100, warmup_ratio=None, image_warmup_epochs=0, image_encoder_frozen=False, save_file="models/clip-model.pt", pivot=False, pivot_alpha=1.0):
    min_val_loss = float("inf")
    if warmup_ratio is not None:
        model.set_freeze_encoders(True)
    if image_warmup_epochs > 0:
        model.set_freeze_encoders(True, encoders="text")
    if image_encoder_frozen:
        model.set_freeze_encoders(True, encoders="image")
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader)
        sum_loss = 0
        pbar.set_description(f"Epoch {epoch}, LR {scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']}")
        if not model.encoders_frozen and epoch == image_warmup_epochs:
            tqdm.write(f"Image encoder is warmed up, thawing text encoders at epoch {epoch}")
            model.set_freeze_encoders(False, encoders="text")
        for batch_idx, (image, *caps) in enumerate(pbar):
            if warmup_ratio is not None:
                if model.encoders_frozen and batch_idx / len(train_loader) >= warmup_ratio:
                    tqdm.write(f"Thawing encoders at step {batch_idx}/{len(train_loader)}")
                    if image_encoder_frozen:
                        model.set_freeze_encoders(False, encoders="text")
                        tqdm.write("Image encoder remaining frozen, thawing text encoders only...")
                    elif image_warmup_epochs == 0:
                        model.set_freeze_encoders(False)
                    elif image_warmup_epochs > 0:
                        model.set_freeze_encoders(False, encoders="image")
            image.pixel_values = image.pixel_values.squeeze(1).to(device)
            image = image
            og_caps = caps
            caps = torch.cat(caps, dim=0).to(device)
            attn_mask = torch.ones_like(caps).float().to(device)
            attn_mask[caps == 0] = 0
            optimizer.zero_grad()
            output = model(image, caps, attn_mask)
            # Align with the pivot image
            loss = F.cross_entropy(output, torch.tile(torch.arange(output.shape[0]//len(train_loader.dataset.langs)), (len(train_loader.dataset.langs),)).to(device))
            if pivot:
                piv = og_caps[0].to(device)
                caps = torch.cat(og_caps[1:], dim=0).to(device)
                attn_mask = torch.ones_like(caps).float().to(device)
                attn_mask[caps == 0] = 0
                # Align with the pivot language
                pivot_attn_mask = torch.ones_like(piv).float().to(device)
                pivot_attn_mask[piv == 0] = 0
                pivot_output = model(piv, caps, attn_mask, pivot_attn_mask=pivot_attn_mask, img_pivot=False)
                loss += pivot_alpha*F.cross_entropy(pivot_output, torch.tile(torch.arange(pivot_output.shape[0]//(len(train_loader.dataset.langs)-1)), (len(train_loader.dataset.langs)-1,)).to(device)) 
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
                    for image, *caps in tqdm(val_loader, desc="Validation", leave=False):
                        image.pixel_values = image.pixel_values.squeeze(1).to(device)
                        image = image
                        caps = torch.cat(caps, dim=0).to(device)
                        attn_mask = torch.ones_like(caps).float().to(device)
                        attn_mask[caps == 0] = 0
                        output = model(image, caps, attn_mask)
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
    argparser.add_argument("--multi30k", action="store_true", help="Use the Multi30k dataset")
    argparser.add_argument("--trans_multi30k", action="store_true", help="Use the translated Multi30k dataset")
    argparser.add_argument("--batch_size", type=int, default=32, help="Set the batch size")
    argparser.add_argument("--learning_rate", type=float, default=1e-5, help="Set the learning rate")
    argparser.add_argument("--warmup_ratio", type=float, default=0.5, help="Set the warmup ratio")
    argparser.add_argument("--image_warmup_epochs", type=int, default=0, help="Set the number of warmup epochs for the image encoder")
    argparser.add_argument("--img_froze", action="store_true", help="Freeze the image encoder")
    argparser.add_argument("--total_epochs", type=int, default=10, help="Set the total number of epochs")
    argparser.add_argument("--pooling", type=str, default="mean", help="Set the pooling type")
    argparser.add_argument("--shuffle_images", action="store_true", help="Shuffle the images")
    argparser.add_argument("--shuffle_val_images", action="store_true", help="Shuffle the validation images")
    argparser.add_argument("--save_file", type=str, default=None, help="Set the save file")
    argparser.add_argument("--pivot_file", type=str, default=None, help="Set the pivot file")
    argparser.add_argument("--piv_alpha", type=float, default=1.0, help="Set the pivot alpha")
    argparser.add_argument("--quechua", action="store_true", help="Include the Quechua captions")
    argparser.add_argument("--name", type=str, default=None, help="Set the name of the run on wandb")
    
    args = argparser.parse_args()
    
    wandb.init(project="clip-parallel", config=args, name=args.name)
    
    batch_size = args.batch_size  # Set the batch size.
    learning_rate = args.learning_rate  # Set the learning rate.
    warmup_ratio = args.warmup_ratio  # Set the warmup ratio.
    image_warmup_epochs = args.image_warmup_epochs
    
    model = ClipModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if args.multilingual:
        train_dataset = MultiLingualCLIPDataset(os.path.join("data", "images", "train2017"), os.path.join("data", "annotations", f"captions_train2017_ordered_translated{'+qu' if args.quechua else ''}.json"), scramble_images=args.shuffle_images, pivot_file=args.pivot_file)
    elif args.fake_multilingual:
        train_dataset = MultiLingualCLIPDataset(os.path.join("data", "images", "train2017"), os.path.join("data", "annotations", "captions_train2017_ordered_fakemulti.json"), scramble_images=args.shuffle_images, pivot_file=args.pivot_file)
    elif args.multilingual_single:
        train_dataset = MultiLingualCLIPDataset(os.path.join("data", "images", "train2017"), os.path.join("data", "annotations", f"captions_train2017_ordered_translated{'+qu' if args.quechua else ''}_singles.json"), scramble_images=args.shuffle_images, pivot_file=args.pivot_file)
    elif args.multi30k:
        train_dataset = MultiLingualCLIPDataset(os.path.join("data", "images", "flickr30k-images"), os.path.join("data", "annotations", "captions_trainmulti30k_ordered_en_de.json"), scramble_images=args.shuffle_images, pivot_file=args.pivot_file)
    elif args.trans_multi30k:
        train_dataset = MultiLingualCLIPDataset(os.path.join("data", "images", "flickr30k-images"), os.path.join("data", "annotations", "captions_trainmulti30k_translated_ordered_en_de.json"), scramble_images=args.shuffle_images, pivot_file=args.pivot_file)
    else:
        train_dataset = MultiLingualCLIPDataset(os.path.join("data", "images", "train2017"), os.path.join("data", "annotations", "captions_train2017_en_only.json"), scramble_images=args.shuffle_images, pivot_file=args.pivot_file)
    # train_dataset = MultiLingualCLIPDataset(os.path.join("data", "images", "train2017"), os.path.join("data", "annotations", "captions_train2017_ordered_translated.json"))
    print(f"Loaded {len(train_dataset)} training examples")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if args.multilingual:
        val_dataset = MultiLingualCLIPDataset(os.path.join("data", "images", "val2017"), os.path.join("data", "annotations", f"captions_val2017_ordered_translated{'+qu' if args.quechua else ''}.json"), scramble_images=args.shuffle_val_images)
    elif args.fake_multilingual:
        val_dataset = MultiLingualCLIPDataset(os.path.join("data", "images", "val2017"), os.path.join("data", "annotations", "captions_val2017_ordered_fakemulti.json"), scramble_images=args.shuffle_val_images)
    elif args.multilingual_single:
        val_dataset = MultiLingualCLIPDataset(os.path.join("data", "images", "val2017"), os.path.join("data", "annotations", f"captions_val2017_ordered_translated{'+qu' if args.quechua else ''}_singles.json"), scramble_images=args.shuffle_val_images)
    elif args.multi30k:
        val_dataset = MultiLingualCLIPDataset(os.path.join("data", "images", "flickr30k-images"), os.path.join("data", "annotations", "captions_valmulti30k_ordered_en_de.json"), scramble_images=args.shuffle_images, pivot_file=args.pivot_file)
    elif args.trans_multi30k:
        val_dataset = MultiLingualCLIPDataset(os.path.join("data", "images", "flickr30k-images"), os.path.join("data", "annotations", "captions_valmulti30k_translated_ordered_en_de.json"), scramble_images=args.shuffle_images, pivot_file=args.pivot_file)
    else:
        val_dataset = MultiLingualCLIPDataset(os.path.join("data", "images", "val2017"), os.path.join("data", "annotations", "captions_val2017_en_only.json"), scramble_images=args.shuffle_val_images)
    print(f"Loaded {len(val_dataset)} validation examples")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_epochs = args.total_epochs  # Set the total number of epochs
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch / total_epochs)
    
    wandb.init(
      # set the wandb project where this run will be logged
      project="DL Project",
    #   name=f"{'multilingual' if args.multilingual else 'english'}-clip-{learning_rate}-{total_epochs}-{warmup_ratio}",
      # track hyperparameters and run metadata
      name=args.name,
      config={
        "lr-init": learning_rate,
        "epochs": total_epochs,
        "batch_size": batch_size,
        "warmup_ratio": warmup_ratio,
      }
    )
    savefile = args.save_file if args.save_file is not None else f"models/{'multi' if args.multilingual else ('fakemulti' if args.fake_multilingual else ('multisingle' if args.multilingual_single else 'en'))}-{ROBERTA_MODEL.split('/')[-1]}-clip{'-pivotaligned'+str(args.piv_alpha) if args.pivot_file is not None else ''}-model-{learning_rate}-{total_epochs}-{warmup_ratio}-imgwarm{image_warmup_epochs}{'imgfroze' if args.img_froze else ''}.pt"
    train(model, device, train_loader, optimizer, epochs=total_epochs, scheduler=scheduler, val_loader=val_loader, val_interval=300, warmup_ratio=warmup_ratio, image_warmup_epochs=image_warmup_epochs, image_encoder_frozen=args.img_froze, save_file=savefile, pivot=args.pivot_file is not None, pivot_alpha=args.piv_alpha)
    wandb.finish()
