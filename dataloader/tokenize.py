meta = "[SPECIES=HUMAN] [TF_CLASS=ZNF]"
seq  = "MDSKLT..."  # full amino-acid sequence
text = meta + " " + seq + " <|endoftext|>"


inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
labels = inputs["input_ids"].clone()
