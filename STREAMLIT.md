# Run TissueLens in Streamlit

The Streamlit interface uses the existing trained model and preprocessing. It caches the model across reruns, keeps results per user session, and omits the evaluation section. Flask remains available through `app.py`.

## Local preview

Run in PowerShell from the `improved` directory:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --timeout 300 --retries 10
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open http://localhost:8501. No training is required.

## Deploy on Streamlit Community Cloud

Prepare a GitHub repository with these files at its root (copy these files from `improved`, keeping the folders shown):

```text
streamlit_app.py
app.py
dataset.py
imaging.py
requirements.txt
.streamlit/config.toml
artifacts/model.keras
artifacts/metadata.json
```

`app.py` is required because the Streamlit interface imports its model loader. It does not start the Flask server when imported. The inference app does not need the training dataset, virtual environment, old checkpoints, or evaluation report.

The existing `.gitignore` excludes `artifacts` and `.keras` files. If using Git to prepare an existing repository, explicitly include only the two required inference files with `git add -f artifacts/model.keras artifacts/metadata.json` (prefix both paths with `improved/` if that is the repository layout). Check staged files before committing. Merely pushing code without the model will leave the hosted app unable to predict.

1. Upload/commit the required files to your chosen GitHub repository.
2. Open https://share.streamlit.io/ and connect the GitHub account that can access it.
3. Choose **Create app**, then select your repository and branch.
4. Set the main file to `streamlit_app.py` (or `improved/streamlit_app.py` if retaining that folder).
5. In advanced settings, select **Python 3.12**, matching the local training environment.
6. Deploy and check the build logs. Once it loads, test an image through the public app URL.

If retaining `improved/`, Community Cloud runs from the repository root; place `.streamlit/config.toml` at the repository root for the theme/upload configuration. Requirements can remain alongside the Streamlit entrypoint.

The repository and trained model have not been published automatically. Cloud deployment still requires your GitHub/Streamlit account. TensorFlow startup can take time; available hosting memory/CPU determines runtime performance.

Official guides: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy and https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization.
