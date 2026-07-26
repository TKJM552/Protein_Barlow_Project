import os
import requests
from Bio.PDB import PDBList

from config import PDB_DIR

#Get initial files from PDB

search_url = "https://search.rcsb.org/rcsbsearch/v2/query"

query_payload = {
    "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "entity_poly.rcsb_entity_polymer_type",
                    "operator": "exact_match",
                    "value": "Protein",
                },
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "entity_poly.rcsb_sample_sequence_length",
                    "operator": "less_or_equal",
                    "value": 1000,
                },
            },
        ],
    },
    "return_type": "entry",
    "request_options": {
        "paginate": {
            "start": 0,
            "rows": 5000  # Return 5,000 PDB IDs (some chains get skipped downstream
                          # if their resolved length exceeds the 1000 cap, so the
                          # processed dataset ends up at or slightly below 5,000)
        }
    },
}


response = requests.post(search_url, json=query_payload)
response.raise_for_status()

#get  list of 4 character PDB IDs from JSON response
data = response.json()
pdb_ids = [hit["identifier"] for hit in data.get("result_set", [])]

print(f"Successfully retrieved {len(pdb_ids)} PDB IDs")

#download the structures via Biopython ($PDB_DIR overrides the destination)
output_dir = PDB_DIR
os.makedirs(output_dir, exist_ok=True)

pdbl = PDBList()
print(f"Downloading files to '{output_dir}'...")

#download_pdb_files handles batch downloading
pdbl.download_pdb_files(pdb_ids, pdir=output_dir, file_format="mmCif")

print("Download complete")

