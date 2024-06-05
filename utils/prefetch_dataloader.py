import threading
import queue
import traceback

class OverlappingDataLoader:
    def __init__(self, dataset, batch_size, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.queue = queue.Queue(maxsize=num_workers * 2)  # Buffer for batches
        self.workers = []
        self.stop_event = threading.Event()
        self.epoch_counter = threading.local()  # Initialize here

        for _ in range(num_workers):
            worker = threading.Thread(target=self._worker_loop)
            worker.start()
            self.workers.append(worker)

    def _worker_loop(self):
        if not hasattr(self.epoch_counter, 'current_epoch'):
            self.epoch_counter.current_epoch = 1
        while not self.stop_event.is_set():
            try:
                epoch = self.epoch_counter.current_epoch
                for i in range(len(self.dataset) // self.batch_size):
                    batch = self.dataset[i * self.batch_size:(i + 1) * self.batch_size]
                    self.queue.put((epoch, batch))
            except Exception as e: 
                print(f"Worker error: {e}")

    def __iter__(self):
        self.epoch_counter = threading.local()  # Thread-local epoch counter
        self.epoch_counter.current_epoch = 1
        return self

    def __next__(self):
        if self.stop_event.is_set():  
            raise StopIteration

        epoch, batch = self.queue.get()
        if epoch != self.epoch_counter.current_epoch:
            self.epoch_counter.current_epoch = epoch
            print(f"Starting epoch {epoch}") 
        return batch

    def stop(self):
        self.stop_event.set()
        for worker in self.workers:
            worker.join()


class PrefetchingDataLoader:
    def __init__(self, dataloader, prefetch_size=2):
        self.dataloader = dataloader
        self.prefetch_size = prefetch_size
        self.queue = queue.Queue(prefetch_size)
        self.dataiter = iter(dataloader)
        self.prefetch_thread = threading.Thread(target=self._prefetch)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def _prefetch(self):
        while True:
            if not self.queue.full():
                try:
                    data = next(self.dataiter)
                    self.queue.put(data)
                except StopIteration:
                    self.queue.put(None)
                    break
                except Exception as e:
                    print(f"Exception in prefetch thread: {e}")
                    print(traceback.format_exc())
                    self.queue.put(None)
                    break

    def __iter__(self):
        return self

    def __next__(self):
        if not self.queue.empty():
            data = self.queue.get()
            if data is None:
                raise StopIteration
            return data
        else:
            raise StopIteration
