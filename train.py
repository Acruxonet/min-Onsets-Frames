import os
import glob
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch import nn, optim
from src import OnsetsAndFrames, PianoDataset, wav_to_mel, load_midi

def train_step(model, batch, optimizer, scaler, criterion):
    mel = batch['mel'].cuda()
    targets = {k: v.cuda() for k, v in batch if k != 'mel'}

    with torch.amp.autocast(device_type='cuda'):
        preds = model(mel)
        loss_onset = criterion(preds['onset_logits'], targets['onset'])
        loss_offset = criterion(preds['offset_logits'], targets['offset'])
        loss_frame = criterion(preds['frame_logits'], targets['frame'])
        mask = targets['onset'] > 0.5
        loss_velocity = torch.mean((preds['velocity_pred'][mask] - targets['velocity'][mask]) ** 2) if mask.sum() > 0 else 0.0

        total_loss = loss_onset + loss_offset + loss_frame + loss_velocity
    optimizer.zero_grad()
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return total_loss.item()

def get_list(data_root):
    audio_files = glob.glob(os.path.join(data_root, "**/*.wav"), recursive=True)
    data_list = []
    for audio in audio_files:
        midi = audio.replace(".wav", ".mid")
        if os.path.exists(midi):
            data_list.append((audio, midi))
            
    return data_list

def main(data_root):
    model = OnsetsAndFrames().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=6e-4)
    scaler = GradScaler(device_type='cuda')
    criterion = nn.BCEWithLogitsLoss()

    train_files = get_list(data_root, split='train')

    dataset = PianoDataset(train_files, segment_seconds=20.0)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

    model.train()
    
    for epoch in range(100):
        epoch_loss = 0
        for i, batch in enumerate(loader):
            loss = train_step(model, batch, optimizer, scaler, criterion)
            epoch_loss += loss
            
            if i % 20 == 0:
                print(f"Epoch {epoch} | Batch {i}/{len(loader)} | Loss: {loss:.4f}")
        
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/piano_model_ep{epoch}.pt")
        print(f"Epoch {epoch} Finished. Loss: {epoch_loss/len(loader):.4f}")

if __name__ == "__main__":
    PATH = "/data" 
    main(PATH)