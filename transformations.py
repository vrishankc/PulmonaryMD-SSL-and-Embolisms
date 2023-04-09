class Transformations(object):
    
    def __init__(self, augmentations, numOfViews = 2):
        self.augmentations = augmentations
        self.numOfViews = numOfViews
    
    def __call__(self, x):
        return [self.augmentations(x) for i in range(self.numOfViews)]
