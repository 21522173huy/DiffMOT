import os
import glob
import numpy as np
from PIL import Image
import torch
from torch import utils
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class DiffMOTDataset(Dataset):
    def __init__(self, path, config=None):
        self.config = config
        self.path = path

        try:
            self.interval = self.config.interval + 1
        except:
            self.interval = 4 + 1

        self.trackers = {}
        self.data = []  

        if os.path.isdir(path):
            if 'MOT' in path:
                self.seqs = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13", "MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
            else:
                self.seqs = [s for s in os.listdir(path) if not s.startswith('.') and "gt_t" not in s]
            self.seqs.sort()
            
            for seq in self.seqs:
                trackerPath = os.path.join(path, seq, "img1/*.txt")
                normalized_path = os.path.normpath(trackerPath) # Normalize path for Windows
                
                self.trackers[seq] = sorted(glob.glob(normalized_path))
                
                for pa in self.trackers[seq]:
                    gt = np.loadtxt(pa, dtype=np.float32)

                    self.precompute_data(seq, gt)  # Precompute data for this sequence




    def precompute_data(self, seq, track_gt):
        """Precompute and store data for the dataset."""
        for init_index in range(len(track_gt) - self.interval):
            cur_index = init_index + self.interval
            cur_gt = track_gt[cur_index]
            cur_bbox = cur_gt[2:6]

            boxes = [track_gt[init_index + tmp_ind][2:6] for tmp_ind in range(self.interval)]
            delt_boxes = [boxes[i+1] - boxes[i] for i in range(self.interval - 1)]
            conds = np.concatenate((np.array(boxes)[1:], np.array(delt_boxes)), axis=1)

            delt = cur_bbox - boxes[-1]

            width, height = cur_gt[7:9]
            image_path = self.path.replace("/trackers_gt_t", "") + f"/{seq}/img1/{int(cur_gt[1]):08d}.jpg"


            data_item = {
                "cur_gt": cur_gt, 
                "cur_bbox": cur_bbox, 
                "condition": conds, 
                "delta_bbox": delt,
                "width": width,
                "height": height,
                "image_path": image_path,
            }

            self.data.append(data_item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    

    def show_image(self, index):
        """Display the image at the given index using PIL."""
        # Get the image path from the dataset
        image_path = self.data[index]['image_path']
        
        # Open the image using PIL
        img = Image.open(image_path)

            
        # Display the image with matplotlib
        plt.imshow(img)
        plt.axis("off")  # Hide axis for a cleaner look
        plt.show()
        
        # Display the image
        img.show()


class DiffMOTDataLoader(utils.data.DataLoader):
    def __init__(self, dataset, config):
        super().__init__(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.preprocess_workers,
            pin_memory=True
        )


