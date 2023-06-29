from models.bubble.data.dataset import BubbleDataset

dataset = BubbleDataset("/home/TUE/s131727/git/bubble-data/n50")

for i in range(8000):
    bubble = dataset[i]
    bubble.as_stl().save(f"stl/{i}.stl")

