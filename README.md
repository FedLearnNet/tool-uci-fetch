# tool-uci-fetch

Fetches a UCI dataset by ID and exports its features and targets as a combined CSV table.

## Runtime inputs

Supply input files according to the schemas in `app.yml`.

## Inputs and outputs

### Inputs

No fields are declared.

### Outputs

| Name | Type | Description |
| --- | --- | --- |
| `report` | TEXT | Small textual report about the fetched dataset and output shape. |
| `dataset` | CSV | Single CSV combining features and targets from the fetched UCI dataset. |

## Configuration

| Name | Default | Description |
| --- | --- | --- |
| `dataset_id` | `17` | UCI dataset numeric id to fetch via ucimlrepo (e.g., 17 = Breast Cancer Wisconsin Diagnostic). |

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
# Set APP_ID and platform URLs before connecting.
python main.py
```

The Docker base image provides the FL-Net API. `requirements.txt` lists additional
runtime dependencies; `requirements-dev.txt` includes the API for local development.
Runtime datasets and trained models must be supplied separately; see `data/README.md`.

Build with `docker build -t tool-uci-fetch .`.
