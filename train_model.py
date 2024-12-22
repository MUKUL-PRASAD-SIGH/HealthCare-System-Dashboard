import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
config = GPT2Config.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)

# Sample data
train_texts = ["Patient has a history of diabetes and hypertension..."]
train_encodings = tokenizer(train_texts, truncation=True, padding=True)

class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = MedicalDataset(train_encodings)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

model.save_pretrained('./model')
tokenizer.save_pretrained('./model')
