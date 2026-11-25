"""AI-assisted Keras v3 configuration. Old checkpoints are intentionally incompatible."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json

PROJECT=Path(__file__).resolve().parents[2]
ROOT=Path(__file__).parent
PREVIOUS=PROJECT.parent/'math_ocr_portable/data/sources'
VERSION='keras-ocr-v6'
GREEK=[r'\Delta',r'\theta',r'\lambda',r'\mu',r'\rho',r'\pi',r'\omega',r'\alpha',r'\beta',r'\gamma',r'\epsilon',r'\sigma',r'\tau',r'\phi',r'\nu']
FUNCTIONS=[r'\ln',r'\log',r'\sin',r'\cos',r'\tan',r'\exp']
CLASSES=list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')+['+','-','=',r'\times','/','(',')','.','[',']',r'\sqrt',r'\pm',r'\cdot',r'\approx',r'\circ']+GREEK
TOKENS=['<PAD>','<BOS>','<EOS>','<UNK>']+CLASSES+['^','_','{','}',r'\frac']+FUNCTIONS
PAD,BOS,EOS,UNK=range(4)
HARD_GLYPHS=(r'\rho','p','O','0',r'\Delta','l','1','I','2','4','3','q','x','X')

@dataclass(frozen=True)
class Settings:
    seed:int=42
    image_size:int=48
    cnn_features:int=256
    projection:int=256
    encoder_hidden:int=256
    encoder_layers:int=2
    embedding:int=128
    dropout:float=.2
    weight_decay:float=1e-4
    batch_size:int=12
    initialization_epochs:int=5
    initialization_steps:int=150
    epochs:int=80
    steps:int=400
    minimum_epochs:int=15
    patience:int=15
    learning_rate:float=3e-4
    min_learning_rate:float=1e-6
    lr_patience:int=4
    lr_factor:float=.5
    symbol_loss_weight:float=.3
    crop_loss_weight:float=.5
    relation_weight:float=.2
    script_relation_boost:float=.3
    attention_weight:float=.15
    clip:float=5.
    generated_train:int=16000
    generated_val:int=800
    generated_test:int=600
    fresh_test:int=300
    composition_train:int=4000
    blind_test:int=300
    transformer_width:int=256
    transformer_heads:int=4
    transformer_layers:int=2
    handwritten_probability:float=.8
    max_symbols:int=80
    max_tokens:int=128
    threads:int=4
    beam_width:int=5
    beam_length_penalty:float=.6
    decode_candidates:tuple=((1,0.),(3,.3),(5,.6))
    log_every:int=20
    relation_radius:float=4.
    merge_epochs:int=15
    merge_batch:int=128
    merge_pairs_per_formula:int=24
    merge_min_precision:float=.98
    merge_max_rounds:int=3

    def validate(self):
        counts=('image_size','cnn_features','projection','encoder_hidden','encoder_layers','embedding','batch_size','initialization_epochs','initialization_steps','epochs','steps','minimum_epochs','patience','lr_patience','generated_train','generated_val','generated_test','max_symbols','max_tokens','threads','log_every','merge_epochs','merge_batch','merge_pairs_per_formula','merge_max_rounds')
        if any(getattr(self,k)<1 for k in counts):raise ValueError('Counts must be positive.')
        if self.minimum_epochs>self.epochs or self.image_size!=48:raise ValueError('Invalid epochs/image size.')
        if min(self.fresh_test,self.composition_train,self.blind_test)<0 or min(self.transformer_width,self.transformer_heads,self.transformer_layers)<1 or self.transformer_width%self.transformer_heads:
            raise ValueError('Invalid held-out count or Transformer dimensions.')
        if not 0<=self.dropout<1 or not 0<=self.handwritten_probability<=1:raise ValueError('Invalid probability.')
        if not 0<self.lr_factor<1 or not 0<self.min_learning_rate<=self.learning_rate:raise ValueError('Invalid learning rate schedule.')
        if any(getattr(self,k)<0 for k in ('symbol_loss_weight','crop_loss_weight','relation_weight','script_relation_boost','attention_weight','weight_decay')):raise ValueError('Negative loss weight.')
        if self.clip<=0 or self.relation_radius<=0 or not 0<self.merge_min_precision<=1:raise ValueError('Invalid clipping/radius/precision.')
        if not self.decode_candidates or any(int(w)!=w or w<1 or a<0 for w,a in self.decode_candidates):raise ValueError('Invalid decoding candidates.')
        return self
    def to_dict(self):return json.loads(json.dumps(asdict(self)))
    @classmethod
    def load(cls,path=None):return cls(**(json.loads(Path(path).read_text(encoding='utf-8-sig')) if path else {})).validate()
