# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

## Image Detection Architectures (Flowcharts)

Below are 4 high-level architecture options you can implement in the prototype. Each flowchart includes where privacy redaction happens and how multiple people / multiple garments are handled.

### 1) Existing Hybrid (YOLO sidecar + optional Vision + Gemini)

```mermaid
flowchart TD
  A[User uploads a photo] --> B[Frontend calls POST /api/items/detect]
  B --> C[Node calls Python vision-sidecar POST /analyze]
  C --> D[Garment + person cues as boxes (and scene tracking)]
  D --> E{worn / selfie scenario?}
  E -->|yes| F[Privacy masking: blur faces using detected face regions]
  E -->|no| G[Skip privacy masking]
  F --> H[Crop each garment bbox -> /storage/processed/*]
  G --> H
  H --> I[Return detected garments list to frontend]
  I --> J[User selects 1..N garments]
  J --> K[For each selected item run processItemPipeline]
  K --> L[POST /api/items/preprocess (pass-through for processed crops)]
  L --> M[POST /api/items/infer (Gemini on the crop; fallback if needed)]
  M --> N[POST /api/items/embed]
  N --> O[POST /api/items/reason (Ollama)]
  O --> P[Item marked Ready in UI]
```

Privacy: faces are blurred before cropping for “worn” photos. Only crops are sent to Gemini/Ollama.

Multi-person / multi-cloth: sidecar returns multiple garment boxes; Node dedupes + ranks, and UI supports selecting multiple crops.

### 2) YOLO-Based Local-First (YOLO-only perception, local privacy)

```mermaid
flowchart TD
  A[User uploads a photo] --> B[Local pipeline (sidecar) runs detection]
  B --> C[Local model(s): person/face detection + garment instance detection]
  C --> D{worn / flat-lay?}
  D -->|worn| E[Local privacy masking: blur/redact faces]
  D -->|no| F[Skip face redaction]
  E --> G[Crop garments (bbox or mask-based) per instance]
  F --> G
  G --> H[If multiple people: assign garment instances to nearest person region]
  H --> I[Return 1..N garment crops]
  I --> J[Frontend/UI selection (optional)]
  J --> K[POST /api/items/infer on garment crops]
  K --> L[POST /api/items/reason (Ollama)]
  L --> M[Ready]
```

Privacy: can be fully local; no cloud vision calls needed for faces.

Quality: strongest when using instance segmentation (better separation when garments overlap).

### 3) Google Vision-Centered (Vision handles detection + privacy, Gemini on crops)

```mermaid
flowchart TD
  A[User uploads a photo] --> B[Node calls Google Vision for face + object cues]
  B --> C[Vision face detection -> face regions]
  C --> D[Privacy masking: blur/redact faces using face boxes]
  D --> E[Vision garment/object localization (or custom-trained Vision detector)]
  E --> F[Crop each detected garment region -> /storage/processed/*]
  F --> G[Return detected garments list to frontend]
  G --> H[User selects 1..N garments]
  H --> I[For each item run processItemPipeline]
  I --> J[POST /api/items/preprocess (pass-through)]
  J --> K[POST /api/items/infer (Gemini on crop; fallback if needed)]
  K --> L[POST /api/items/embed]
  L --> M[POST /api/items/reason (Ollama)]
  M --> N[Ready]
```

Privacy: faces are redacted immediately after Vision returns face boxes; Gemini only sees garment crops.

Multi-person / multi-cloth: depends on Vision garment localization quality (best with a custom garment detector).

### 4) Open-Vocabulary + SAM (prompt boxes -> masks; mask-based crops)

```mermaid
flowchart TD
  A[User uploads a photo] --> B[Local open-vocabulary detector proposes regions]
  B --> C[GroundingDINO/OWOD outputs boxes for labels: person, shirt, pants, dress...]
  C --> D[Run SAM to convert boxes into high-quality masks]
  D --> E{privacy redaction needed? (person present)}
  E -->|yes| F[Local privacy masking: blur/redact face/person mask regions]
  E -->|no| G[Keep as-is]
  F --> H[Mask-based garment crops (extract each garment instance)]
  G --> H
  H --> I[If multiple people: associate garment masks to each person mask by overlap]
  I --> J[Return 1..N garment instances/masks]
  J --> K[Frontend selection (optional)]
  K --> L[POST /api/items/infer (Gemini on mask-cropped garment)]
  L --> M[POST /api/items/reason (Ollama)]
  M --> N[Ready]
```

Privacy: can remain local; cloud is only used for Gemini/Ollama on already-redacted crops.

Quality: typically highest for overlaps and clutter because masks are more precise than bounding boxes.

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
