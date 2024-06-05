import pickle
import os


# # Function to initialize the cache
# def initialize_cache(dataloader, cache_file, cache_epochs):
#     with open(cache_file, "wb") as f:
#         for epoch in range(cache_epochs):
#             epoch_batches = []
#             for batch in dataloader:
#                 print(
#                     f"Init: Epoch {epoch + 1} Batch {len(epoch_batches) + 1}/{len(dataloader)}",
#                     flush=True,
#                 )
#                 epoch_batches.append(batch)
#             pickle.dump(epoch_batches, f)


# Function to initialize the cache
def initialize_cache(dataloader, cache_dir, cache_epochs):
    os.makedirs(cache_dir, exist_ok=True)
    for epoch in range(cache_epochs):
        epoch_batches = []
        for batch in dataloader:
            print(
                f"Init: Epoch {epoch + 1} Batch {len(epoch_batches) + 1}/{len(dataloader)}",
                flush=True,
            )
            epoch_batches.append(batch)
        with open(os.path.join(cache_dir, f"epoch_{epoch}.pkl"), "wb") as f:
            pickle.dump(epoch_batches, f)


# # Function to append batches to the cache file
# def append_batches_to_cache(dataloader, cache_file, current_epoch):
#     epoch_batches = []
#     for batch in dataloader:
#         print(
#             f"Append: Epoch {current_epoch} Batch {len(epoch_batches) + 1}/{len(dataloader)}",
#             flush=True,
#         )
#         epoch_batches.append(batch)

#     with open(cache_file, "ab") as f:
#         pickle.dump(epoch_batches, f)


# Function to append batches to the cache file
def append_batches_to_cache(dataloader, cache_dir, current_epoch):
    epoch_batches = []
    for batch in dataloader:
        print(
            f"Append: Epoch {current_epoch} Batch {len(epoch_batches) + 1}/{len(dataloader)}",
            flush=True,
        )
        epoch_batches.append(batch)

    with open(
        os.path.join(cache_dir, f"epoch_{current_epoch}.pkl"), "wb"
    ) as f:  # Save each epoch to a separate file
        pickle.dump(epoch_batches, f)


# # Function to load a specific epoch's cached batches
# def load_cached_batches(cache_file, epoch_index):
#     with open(cache_file, "rb") as f:
#         for _ in range(epoch_index + 1):
#             epoch_batches = pickle.load(f)
#     return epoch_batches


# Function to load a specific epoch's cached batches
def load_cached_batches(cache_dir, epoch_index):
    with open(os.path.join(cache_dir, f"epoch_{epoch_index}.pkl"), "rb") as f:
        epoch_batches = pickle.load(f)
    return epoch_batches


# # Custom DataLoader wrapper
# class EpochCachingDataLoader:
#     def __init__(self, dataloader, cache_file, cache_epochs):
#         self.dataloader = dataloader
#         self.cache_file = cache_file
#         self.cache_epochs = cache_epochs
#         self.current_epoch = 0

#         if not os.path.exists(cache_file):
#             print("Caching initial batches...")
#             initialize_cache(dataloader, cache_file, cache_epochs)

#     def __iter__(self):
#         if self.current_epoch < self.cache_epochs:
#             print(f"Using cached data for epoch {self.current_epoch}")
#             self.epoch_batches = iter(
#                 load_cached_batches(self.cache_file, self.current_epoch)
#             )
#         else:
#             print(f"Using DataLoader for epoch {self.current_epoch}")
#             append_batches_to_cache(
#                 self.dataloader, self.cache_file, self.current_epoch
#             )
#             self.epoch_batches = iter(
#                 load_cached_batches(self.cache_file, self.current_epoch)
#             )

#         self.current_epoch += 1
#         return self

#     def __next__(self):
#         return next(self.epoch_batches)


# Custom DataLoader wrapper
class EpochCachingDataLoader:
    def __init__(self, dataloader, cache_dir, current_epoch=0, cache_epochs=0):
        self.dataloader = dataloader
        self.cache_dir = cache_dir
        self.current_epoch = current_epoch

        if not os.path.exists(cache_dir):
            print("Caching initial batches...")
            initialize_cache(dataloader, cache_dir, cache_epochs)

    def __iter__(self):
        cache_available = os.path.isfile(
            os.path.join(self.cache_dir, f"epoch_{self.current_epoch}.pkl")
        )
        if cache_available:
            print(f"Using cached data for epoch {self.current_epoch}")
            self.epoch_batches = iter(
                load_cached_batches(self.cache_dir, self.current_epoch)
            )
        else:
            print(f"Using DataLoader for epoch {self.current_epoch}")
            append_batches_to_cache(self.dataloader, self.cache_dir, self.current_epoch)
            self.epoch_batches = iter(
                load_cached_batches(self.cache_dir, self.current_epoch)
            )

        self.current_epoch += 1
        return self

    def __next__(self):
        return next(self.epoch_batches)

    def __len__(self):
        return len(self.dataloader)


# # Function to cache batches
# def cache_batches(dataloader, cache_file, num_epochs, append=False):
#     batches = []

#     # Load existing cache if appending
#     if append and os.path.exists(cache_file):
#         with open(cache_file, "rb") as f:
#             batches = pickle.load(f)

#     for epoch in range(num_epochs):
#         epoch_batches = []
#         for batch in dataloader:
#             print(f"Batch {len(epoch_batches) + 1}/{len(dataloader)}", flush=True)
#             epoch_batches.append(batch)
#         batches.append(epoch_batches)

#     with open(cache_file, "wb") as f:
#         pickle.dump(batches, f)


# # Function to load cached batches
# def load_cached_batches(cache_file):
#     with open(cache_file, "rb") as f:
#         return pickle.load(f)


# # Custom DataLoader wrapper
# class EpochCachingDataLoader:
#     def __init__(self, dataloader, cache_file, cache_epochs):
#         self.dataloader = dataloader
#         self.cache_file = cache_file
#         self.cache_epochs = cache_epochs
#         self.current_epoch = 0
#         self.cached_batches = None

#         if not os.path.exists(cache_file):
#             print("Caching initial batches...")
#             cache_batches(dataloader, cache_file, cache_epochs)

#         self.cached_batches = load_cached_batches(cache_file)

#     def __iter__(self):
#         if self.current_epoch < len(self.cached_batches):
#             print(f"Using cached data for epoch {self.current_epoch}")
#             self.epoch_batches = iter(self.cached_batches[self.current_epoch])
#         else:
#             print(f"Using DataLoader for epoch {self.current_epoch}")
#             self.epoch_batches = iter(self.dataloader)

#             # Append new batches to cache
#             new_batches = []
#             for batch in self.epoch_batches:
#                 new_batches.append(batch)

#             self.cached_batches.append(new_batches)

#             with open(self.cache_file, "wb") as f:
#                 pickle.dump(self.cached_batches, f)

#             self.epoch_batches = iter(new_batches)

#         self.current_epoch += 1
#         return self

#     def __next__(self):
#         return next(self.epoch_batches)
