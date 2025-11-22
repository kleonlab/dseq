prompt = "[SPECIES=HUMAN] [TF_CLASS=HOMEOBOX]"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
gen_ids = model.generate(
    input_ids,
    max_length=600,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    num_return_sequences=10,
    eos_token_id=tokenizer.eos_token_id
)

for ids in gen_ids:
    seq = tokenizer.decode(ids, skip_special_tokens=True)
    print(seq)
