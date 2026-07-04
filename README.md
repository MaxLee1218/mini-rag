# mini-rag
my first RAG project

## Environment Variables

Create a `.env` file in the project root using `.env.example` as a template:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_TIMEOUT=30
```

The `.env` file is ignored by Git. The project automatically loads it through
`python-dotenv` when `app.config` is imported.
