from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch

class FlowersDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform or v2.Compose([
                v2.Resize((224, 224)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def get_labels(self):
        return self.data['label'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]['image']
        label = torch.tensor(self.data.iloc[idx]['label'], dtype=torch.long)

        try:
            image = Image.open(image_path).convert('RGB')
        except (OSError, UnidentifiedImageError):
            print('Could not read image:', image_path)
            next_idx = idx + 1
            return self.__getitem__(next_idx)

        image = self.transform(image)
        return image, label