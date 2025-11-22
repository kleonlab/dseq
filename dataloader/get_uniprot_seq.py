import time
from io import StringIO

import requests
import pandas as pd

def map_ensembl_to_uniprot(ensembl_ids, from_db="Ensembl", to_db="UniProtKB"):
    """
    Given a list of Ensembl gene IDs, map them to UniProtKB accessions using UniProt API.
    Returns a pandas DataFrame with columns: ensembl_id, uni_prot_accession, status, etc.
    """
    # 1. Submit job
    url_submit = "https://rest.uniprot.org/idmapping/run"
    params = {
        "from": from_db,
        "to": to_db,
        "ids": ",".join(ensembl_ids)
    }
    resp = requests.post(url_submit, data=params)
    resp.raise_for_status()
    job_id = resp.json()["jobId"]

    # 2. Poll job status
    url_status = f"https://rest.uniprot.org/idmapping/status/{job_id}"
    while True:
        resp = requests.get(url_status)
        resp.raise_for_status()
        payload = resp.json()
        status = payload.get("jobStatus") or payload.get("status")
        if not status:
            # API sometimes returns `messages` when it cannot provide status yet.
            raise RuntimeError(f"Unable to determine job status: {payload}")
        if status == "FINISHED":
            break
        if status in ("RUNNING", "NEW"):
            time.sleep(1)
            continue
        # Propagate any error details provided by the API.
        raise RuntimeError(f"Mapping job failed: {payload}")

    # 3. Retrieve results
    url_results = f"https://rest.uniprot.org/idmapping/uniprotkb/results/{job_id}"
    resp = requests.get(url_results, params={"format":"tsv"})
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), sep="\t")
    return df

def fetch_fasta_for_accessions(accessions):
    """
    Given a list of UniProt accessions, fetch the FASTA sequences.
    Returns dict: accession -> sequence (string)
    """
    fasta_dict = {}
    batch_size = 100
    for i in range(0, len(accessions), batch_size):
        batch = accessions[i:i+batch_size]
        url_fasta = "https://rest.uniprot.org/uniprotkb/stream"
        params = {
            "format": "fasta",
            "accessions": ",".join(batch)
        }
        resp = requests.get(url_fasta, params=params)
        resp.raise_for_status()
        fasta_text = resp.text
        # parse FASTA
        for record in fasta_text.strip().split(">")[1:]:
            header, *seq_lines = record.split("\n")
            accession = header.split("|")[1]  # e.g., sp|P12345-1|...
            sequence = "".join(seq_lines)
            fasta_dict[accession] = sequence
    return fasta_dict

# Example use:
ensembl_ids = ["ENSG00000137203"]  # example human gene IDs
mapping_df = map_ensembl_to_uniprot(ensembl_ids)
print(mapping_df.head())

uni_accessions = mapping_df["To"].unique().tolist()
fasta_dict = fetch_fasta_for_accessions(uni_accessions)
print({acc: len(seq) for acc, seq in fasta_dict.items()})

# Search fasta_dict for isoform sequences (accessions with a dash in UniProt convention, e.g. "P12345-2")
isoform_seqs = {acc: seq for acc, seq in fasta_dict.items() if "-" in acc}
print("Isoform sequences found:", {acc: len(seq) for acc, seq in isoform_seqs.items()})
