#TODO: Add 2nd GCN or try different GNN Layer
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch_geometric.datasets import Planetoid
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import ModifiedLlamaForCausalLM
import shutil
import matplotlib.pyplot as plt

# Clear GPU memory
torch.cuda.empty_cache()

# Redownload Cora dataset
shutil.rmtree('./data/Planetoid/Cora', ignore_errors=True)
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

# Verify dataset
print(f"Cora dataset saved at: {dataset.root}")

# Extract and normalize node features
node_features = data.x  # [2708, 1433]
node_features = (node_features - node_features.mean(dim=0)) / (node_features.std(dim=0) + 1e-8)
edge_index = data.edge_index  # [2, 10556]
labels = data.y  # [2708]
train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask

# Verify shapes
print(f"Node features: {node_features.shape}, Edge index: {edge_index.shape}, Labels: {labels.shape}")

# Move tensors to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
node_features = node_features.to(device)
edge_index = edge_index.to(device)
labels = labels.to(device)
train_mask = train_mask.to(device)
val_mask = val_mask.to(device)
test_mask = test_mask.to(device)

# Configure LLaMA
config = LlamaConfig(
    hidden_size=1433,
    num_hidden_layers=4,
    num_attention_heads=11,
    intermediate_size=1024,
    max_position_embeddings=2708,
    vocab_size=0,  # No embedding layer
    output_hidden_states=True
)
model = ModifiedLlamaForCausalLM(config)
model.to(device)

# Classification head
classifier = nn.Linear(config.hidden_size, dataset.num_classes).to(device)
optimizer = AdamW(list(model.parameters()) + list(classifier.parameters()), lr=0.0002, weight_decay=0.01)  # Adjusted LR
criterion = CrossEntropyLoss()

# Training loop
model.train()
val_accs, val_losses = [], []
for epoch in range(100):
    optimizer.zero_grad()
    
    outputs = model(
        inputs_embeds=node_features.unsqueeze(0),  # [1, 2708, 1433]
        edge_index=edge_index,
        return_dict=True,
        output_hidden_states=True
    )
    hidden_states = outputs.hidden_states
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[-1]
    hidden_states = hidden_states.squeeze(0)  # [2708, 1433]
    logits = classifier(hidden_states)
    loss = criterion(logits[train_mask], labels[train_mask])
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Validation
    with torch.no_grad():
        val_outputs = model(
            inputs_embeds=node_features.unsqueeze(0),
            edge_index=edge_index,
            return_dict=True,
            output_hidden_states=True
        )
        val_hidden_states = val_outputs.hidden_states
        if isinstance(val_hidden_states, tuple):
            val_hidden_states = val_hidden_states[-1]
        val_hidden_states = val_hidden_states.squeeze(0)
        val_logits = classifier(val_hidden_states)[val_mask]
        val_loss = criterion(val_logits, labels[val_mask])
        val_pred = val_logits.argmax(dim=1)
        val_accuracy = (val_pred == labels[val_mask]).float().mean()

        val_losses.append(val_loss.item())
        val_accs.append(val_accuracy.item())
    
    print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.2f}")

# Standard evaluation
model.eval()
with torch.no_grad():
    outputs = model(
        inputs_embeds=node_features.unsqueeze(0),
        edge_index=edge_index,
        return_dict=True,
        output_hidden_states=True
    )
    hidden_states = outputs.hidden_states
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[-1]
    hidden_states = hidden_states.squeeze(0)
    logits = classifier(hidden_states)
    pred = logits.argmax(dim=1)
    accuracy = (pred[test_mask] == labels[test_mask]).float().mean()
    print(f"Test Accuracy: {accuracy:.2f}")

plt.plot(val_accs, label='Val Accuracy')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.savefig("./training_plot.png")