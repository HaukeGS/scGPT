class StreamingDataLoaderWrapper:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.current_batch = 0
        self.is_last_batch = False
    
    def __iter__(self):
        self.current_batch = 0
        self.iterator = iter(self.dataloader)
        self.next_batch = None
        
        # Pre-fetch the first batch
        try:
            self.next_batch = next(self.iterator)
        except StopIteration:
            return
        
        return self
    
    def __next__(self):
        if self.next_batch is None:
            raise StopIteration
        
        current_data = self.next_batch
        
        # Try to get the next batch to see if current is last
        try:
            self.next_batch = next(self.iterator)
            self.is_last_batch = False
        except StopIteration:
            self.next_batch = None
            self.is_last_batch = True
        
        result = {
            'data': current_data,
            'batch_idx': self.current_batch,
            'is_last_batch': self.is_last_batch
        }
        
        self.current_batch += 1
        return result