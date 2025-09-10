### Hugging Face Token Note

This package includes a `.env.local` file next to the executable that temporarily contains the author's Hugging Face access token so a teacher can run the app without setup.

What this means:
- This build will authenticate to Hugging Face automatically and can access gated models.
- For long‑term use, you should create your own token and replace the temporary one.

Replace the token (recommended):
1) Open the app folder (same folder as `ScaffoldAI-EnhancedUI.exe`).
2) Open `.env.local` in a text editor.
3) Replace the value after `HUGGINGFACE_TOKEN=` with your own token, e.g.:
   ```dotenv
   HUGGINGFACE_TOKEN=hf_YourOwnPersonalTokenHere
   ```
4) Save the file and re‑launch the app.

Get your token:
- Sign in at `https://huggingface.co/settings/tokens` → Create a token with “Read” scope.
- Some models may require you to accept a license on their model page.

Security & scope:
- Do not share or commit your personal token to version control.
- You can revoke tokens anytime on the tokens page above.

Without a token:
- The app can still run, but model access may be limited and downloads may fail for gated models. Add a token to avoid these issues.


