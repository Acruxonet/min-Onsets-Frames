from constant import *
from modules import *

class FrameFusion(nn.Module):
    def __init__(self, output_features=88, model_size=768):
        super().__init__()
        self.lstm = BiLSTM(output_features * 3, model_size)
        self.linear = nn.Linear(model_size, output_features)

    def forward(self, onset_logits, offset_logits, frame_logits):
        onset_probs = torch.sigmoid(onset_logits).detach()
        offset_probs = torch.sigmoid(offset_logits).detach()
        frame_probs = torch.sigmoid(frame_logits)

        combined = torch.cat([onset_probs, offset_probs, frame_probs], dim = -1)
        x = self.lstm(combined)
        return self.linear(x)
    
class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features=229, output_features=88, model_size=768):
        def build_pipeline(is_bilstm=True):
            layers = [
                AcousticExtractor(input_features, out_features=model_size)
            ]
            if is_bilstm:
                layers.append(BiLSTM(model_size, model_size // 2))
            layers.append(nn.Linear(model_size, output_features))
            return nn.Sequential(*layers)
        
        self.onset_stack = build_pipeline(is_bilstm=True)
        self.offset_stack = build_pipeline(is_bilstm=True)
        self.frame_stack = build_pipeline(is_bilstm=False)
        self.velocity_stack = build_pipeline(is_bilstm=False)
        self.frame_fusion = FrameFusion(output_features, model_size)
        
    def forward(self, x):
        onset_logits = self.onset_stack(x)
        offset_logits = self.offset_stack(x)
        frame_logits = self.frame_stack(x)
        velocity_pred = self.velocity_stack(x)

        activation_logits = self.frame_fusion(onset_logits, offset_logits, frame_logits)
        return {
            'onset_logits': onset_logits,
            'offset_logits': offset_logits,
            'activation_logits': activation_logits,
            'frame_logits': frame_logits,
            'velocity_pred': velocity_pred
        }