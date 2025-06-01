import torch
import torch.nn as nn
import os
import json
from PIL import Image
from tqdm import tqdm
from transformers import XLMRobertaModel, ViTModel, ViTImageProcessor, AutoTokenizer

ROBERTA_MODEL = "xlm-roberta-large"
VIT_MODEL = "google/vit-base-patch16-224-in21k"

class MultiLingualCLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, caption_file):
        self.images = {}
        for image_file in os.listdir(image_dir):
            index = int(image_file.split(".")[0])
            # with Image.open(os.path.join(image_dir, image_file)) as img:
            #     self.images[index] = img.copy()
            self.images[index] = os.path.join(image_dir, image_file)
        self.tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
        self.image_processor = ViTImageProcessor.from_pretrained(VIT_MODEL)
        with open(caption_file, "r") as f:
            self.captions = json.load(f)
        self.langs = list(self.captions[0]["captions"].keys())
        # self.max_len = max([max([len(self.tokenizer.encode(caption["captions"][lang])) for caption in self.captions]) for lang in self.langs])
        self.max_len = 128
        


    def __len__(self):
        return len(self.images)
    
    def preprocess_text(self, text):
        return torch.Tensor(self.tokenizer.encode(text, padding="max_length", max_length=self.max_len)).long()
    
    def __getitem__(self, idx):
        cap_dict = self.captions[idx]
        caps = [self.preprocess_text(cap_dict["captions"][lang]) for lang in self.langs]
        with Image.open(self.images[cap_dict["image_id"]]) as img:
            image = self.image_processor.preprocess(img.convert("RGB"), return_tensors="pt")
        # image = self.image_processor(self.images[cap_dict["image_id"]], return_tensors="pt")
        return image, *caps

class TextOnlyCLIPDataset(torch.utils.data.Dataset):  
    def __init__(self, caption_file):
        self.tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
        with open(caption_file, "r") as f:
            self.captions = json.load(f)
        self.langs = list(self.captions[0]["captions"].keys())
        self.max_len = 128
        
    def __len__(self):
        return len(self.captions)
    
    def preprocess_text(self, text):
        return torch.Tensor(self.tokenizer.encode(text, padding="max_length", max_length=self.max_len)).long()
    
    def __getitem__(self, idx):
        cap_dict = self.captions[idx]
        caps = [self.preprocess_text(cap_dict["captions"][lang]) for lang in self.langs]
        return None, *caps

class ClipModel(nn.Module):
    def __init__(self):
        super(ClipModel, self).__init__()
        # Define your layers here
        # Encoders
        self.text_encoder = XLMRobertaModel.from_pretrained(ROBERTA_MODEL)
        self.image_encoder = ViTModel.from_pretrained(VIT_MODEL)
        self.text_dense = nn.Linear(self.text_encoder.config.hidden_size, 512)
        self.image_dense = nn.Linear(self.image_encoder.config.hidden_size, 512)
        self.temp = nn.Parameter(torch.ones(1))
        self.encoders_frozen = False
        
    def encode_text(self, cap, attn_mask=None):
        return self.text_dense(self.text_encoder(cap, attention_mask=attn_mask).last_hidden_state[:, 0, :]).detach()
    
    def encode_image(self, img):
        if img is None:
            return torch.zeros(512)
        
        return self.image_dense(self.image_encoder(pixel_values=img.pixel_values).last_hidden_state[:, 0, :]).detach()


def generate_vectors(model, data_loader, device):
    model.eval()
    langs = data_loader.dataset.langs
    text_vectors = {lang: [] for lang in langs}
    image_vectors = []
    with torch.no_grad():
        for (image, *caps) in tqdm(data_loader):
            image.pixel_values = image.pixel_values.squeeze(1).to(device)
            caps = [cap.to(device) for cap in caps]
            attn_masks = [(torch.ones_like(cap) * (cap != 0).float()).float().to(device) for cap in caps]
            image_vec = model.encode_image(image)
            cap_vecs = [model.encode_text(cap, attn_mask) for cap, attn_mask in zip(caps, attn_masks)]
            for lang, cap_vec in zip(langs, cap_vecs):
                # unpack the batch dimension and append to the list
                text_vectors[lang].extend(cap_vec.tolist())
            image_vectors.extend(image_vec.tolist())
    
    return image_vectors, text_vectors

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="data/images/val2017")
    parser.add_argument("--caption_file", type=str, default="data/annotations/captions_val2017_ordered_translated_with_unseen.json")
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--text_only", action="store_true")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join("vectors", os.path.basename(args.model_file))    
    os.makedirs(args.output_dir, exist_ok=True)
    
    
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MultiLingualCLIPDataset(args.image_dir, args.caption_file)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = ClipModel()
    model.load_state_dict(torch.load(args.model_file))
    model = model.to(device)
    print("Generating vectors...")
    image_vectors, text_vectors = generate_vectors(model, data_loader, device)

    image_vec_file = os.path.join(args.output_dir, "image_vectors.tsv")
    text_vec_files = {lang: os.path.join(args.output_dir, f"{lang}_text_vectors.tsv") for lang in text_vectors.keys()}
    
    
    print(f"Writing vectors to files: {image_vec_file}, {', '.join(text_vec_files.values())}")
    with open(image_vec_file, "w") as f:
        for vec in image_vectors:
            f.write("\t".join([str(v) for v in vec]) + "\n")
    for lang, vec_file in text_vec_files.items():
        with open(vec_file, "w") as f:
            for vec in text_vectors[lang]:
                f.write("\t".join([str(v) for v in vec]) + "\n")