# Fake-detection
## To start  
1. Create data folder in Fake-detection folder
2. Download datasets 1-4 into it. 
3. Run compressor.py to create compressed versions of datasets  
Now You can use data loader for loading data from datasets  
## DataLoader
You can use load function with parameters:  
1. dataset_nr - number of your dataset (1-4)
2. samples_amount - amount of samples to load, the samples are balanced (real and fake)
3. shuffle=False - if True takes random samples_amount of samples, otherwise takes first samples_amount
4. compressed=False - if True takes compressed pictures otherwise orginal 
