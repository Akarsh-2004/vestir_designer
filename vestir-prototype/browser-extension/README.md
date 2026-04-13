# Vestir Browser Extension (MVP)

This extension lets you open any shopping/product page, pick the page's main clothing image, and match it against your closet items already embedded by the Vestir API.

## What it does

- Extracts the current page title + best product image URL.
- Sends that image to `POST /api/extension/match`.
- Backend runs preprocess + infer + embedding and returns top similar closet items.

## Prerequisites

- Run Vestir API locally (`npm run dev:api` in `vestir-prototype`).
- Add at least a few closet items through the app so `/api/items/embed` has vectors in `server/storage/embeddings.json`.

## Load in Chrome

1. Go to `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select this folder: `vestir-prototype/browser-extension`

## Usage

1. Open an e-commerce page with a clothing product image.
2. Open the extension popup.
3. Set API base URL (default `http://127.0.0.1:8787`).
4. Click **Match Current Product**.
