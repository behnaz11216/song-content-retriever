\# Song Content Retriever



An experimental Python project that compares text with safety-related questions using BERT embeddings.



\## Demo input



The script uses synthetic demo text defined in `song\_retriever.py`. It does not include a collection of real song lyrics.



\## Setup



Create and activate a virtual environment, then install dependencies:



```powershell

python -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt



```



Run the script:



```bash

python song\_retriever.py

```



The first run downloads the `bert-base-uncased` model.



\## Limitations



This is an experiment, not a validated child-safety classifier. Similarity scores and labels may be wrong. Unclear results should be reviewed by a person; do not rely on this script alone to make decisions about children.

