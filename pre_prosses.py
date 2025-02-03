import torchaudio
from torch.utils.data import Dataset, DataLoader

#initial download
# SC_dataset = torchaudio.datasets.SPEECHCOMMANDS(
#     root= '.',
#      download=True , subset = "training")



SC_dataset = torchaudio.datasets.SPEECHCOMMANDS(
    root= '.',
     download=False , subset = "training")
SC_loader = DataLoader(SC_dataset,
    batch_size=1,
    shuffle=True )
