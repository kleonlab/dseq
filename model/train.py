from loader_data import get_dataloader

datapath = "/home/u5bc/sanjukta.u5bc/dseq/datasets/fasta_data"
dataloader = get_dataloader(datapath, batch_size=16)

for batch in dataloader:
    # batch['input_ids']  # batched tensor
    # batch['labels']     # batched tensor
    # batch['entry']      # list of strings
    # batch['metadata']   # list of dicts (if metadata loaded)

    #print(batch['entry'])
    #print(batch['labels'])
    #print(batch['metadata'])
    #print(batch['input_ids'])
    pass