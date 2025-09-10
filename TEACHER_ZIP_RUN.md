### ScaffoldAI — Super Simple Instructions (no Python needed)

Do these steps exactly:
1) Download the zip you were sent (for example `ScaffoldAI-EnhancedUI-vX.Y.Z.zip`).
2) Right‑click the zip → Properties → check “Unblock” (if you see it) → Apply → OK.
3) Right‑click the zip → Extract All… → choose a simple place like `C:\ScaffoldAI\`.
4) Open the folder `ScaffoldAI-EnhancedUI\` you just extracted.
5) Double‑click `ScaffoldAI-EnhancedUI.exe`.
   - If Windows shows a blue warning, click “More info” → “Run anyway”.
6) In your browser, open `http://localhost:5002`. If that doesn’t load, try `http://localhost:5003`.

Optional (only if you were given a token for more models):
Create a file named `.env.local` next to the `.exe` with this one line, then run the app again:
```dotenv
HUGGINGFACE_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

If something goes wrong:
- “python312.dll” error: install this and try again: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Model locked/needs access: add the token above, or accept the model license on Hugging Face.
- Blank page: make sure the folder `vector_outputs` is inside `ScaffoldAI-EnhancedUI\` and contains:
  - `scaffold_index_1.faiss`
  - `scaffold_metadata_1.json`

---