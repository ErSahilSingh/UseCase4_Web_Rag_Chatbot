import axios from "axios";
import * as cheerio from "cheerio";
import readline from "readline";
import OpenAI from "openai";
import { ChromaClient } from "chromadb";
import dotenv from "dotenv";
dotenv.config();

// ─── Setup ────────────────────────────────────────────────────────────────────

const openai = new OpenAI();

const chroma = new ChromaClient({
  path: process.env.CHROMA_URL || "http://localhost:8000",
});

const COLLECTION_NAME = "web_rag";
let collection;

async function initChroma() {
  // Delete old collection if it exists so each run starts fresh
  try {
    await chroma.deleteCollection({ name: COLLECTION_NAME });
  } catch (_) {
    // Didn't exist yet — that's fine
  }
  collection = await chroma.createCollection({
    name: COLLECTION_NAME,
    metadata: { "hnsw:space": "cosine" },
  });
  console.log("🗄️  ChromaDB collection ready.\n");
}

// ─── Scraper ──────────────────────────────────────────────────────────────────

async function scrape(url) {
  console.log(`\n🌐 Scraping ${url} ...`);
  const { data } = await axios.get(url, {
    headers: { "User-Agent": "Mozilla/5.0 (compatible; RAG-bot/1.0)" },
    timeout: 15_000,
  });

  const $ = cheerio.load(data);

  // Remove noise
  $("script, style, noscript, svg, img").remove();

  const title = $("title").text().trim();
  const bodyText = $("body").text().replace(/\s+/g, " ").trim();

  const internalLinks = [];
  const externalLinks = [];

  $("a[href]").each((_, el) => {
    const href = $(el).attr("href");
    if (!href || href === "/" || href.startsWith("#")) return;
    if (href.startsWith("http://") || href.startsWith("https://")) {
      externalLinks.push(href);
    } else {
      internalLinks.push(href);
    }
  });

  return { title, bodyText, internalLinks, externalLinks };
}

// ─── Chunking ─────────────────────────────────────────────────────────────────

function chunkText(text, maxWords = 200) {
  const words = text.split(/\s+/).filter(Boolean);
  const chunks = [];

  for (let i = 0; i < words.length; i += maxWords) {
    chunks.push(words.slice(i, i + maxWords).join(" "));
  }

  return chunks;
}

// ─── Embeddings ───────────────────────────────────────────────────────────────

async function embed(text) {
  const res = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
    encoding_format: "float",
  });
  return res.data[0].embedding;
}

// ─── Ingest ───────────────────────────────────────────────────────────────────

async function ingest(url) {
  const { title, bodyText, internalLinks, externalLinks } = await scrape(url);

  console.log(`📄 Title : ${title || "(none)"}`);
  console.log(`🔗 Internal links: ${internalLinks.length}  External: ${externalLinks.length}`);

  const chunks = chunkText(bodyText);
  console.log(`✂️  Splitting into ${chunks.length} chunks, embedding & storing in Chroma...`);

  // Batch upsert in groups of 50 to avoid large payloads
  const BATCH = 50;
  for (let i = 0; i < chunks.length; i += BATCH) {
    const batch = chunks.slice(i, i + BATCH);
    const embeddings = await Promise.all(batch.map(embed));

    await collection.add({
      ids: batch.map((_, j) => `chunk-${i + j}`),
      embeddings,
      documents: batch,
      metadatas: batch.map(() => ({ source: url })),
    });

    process.stdout.write(`\r   Stored ${Math.min(i + BATCH, chunks.length)}/${chunks.length} chunks `);
  }

  console.log("\n✅ Ingestion complete!\n");
}

// ─── Retrieval ────────────────────────────────────────────────────────────────

async function retrieve(query, topK = 4) {
  const queryEmbedding = await embed(query);

  const results = await collection.query({
    queryEmbeddings: [queryEmbedding],
    nResults: topK,
  });

  // results.documents is [[doc1, doc2, ...]]
  return results.documents[0] ?? [];
}

// ─── Chat ─────────────────────────────────────────────────────────────────────

/** Full conversation history for multi-turn context */
const messages = [];

async function chat(userMessage) {
  const relevantChunks = await retrieve(userMessage);
  const context = relevantChunks.join("\n\n---\n\n");

  // System prompt injected fresh each turn (not stored in history)
  const systemPrompt = `You are a helpful assistant. Answer questions using ONLY the context below.
If the context doesn't contain enough information, say so honestly.

CONTEXT:
${context}`;

  const response = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: systemPrompt },
      ...messages,
      { role: "user", content: userMessage },
    ],
  });

  const assistantMessage = response.choices[0].message.content;

  // Store for multi-turn memory
  messages.push({ role: "user", content: userMessage });
  messages.push({ role: "assistant", content: assistantMessage });

  return assistantMessage;
}

// ─── Terminal UI ──────────────────────────────────────────────────────────────

function prompt(rl, question) {
  return new Promise((resolve) => rl.question(question, resolve));
}

async function main() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  console.log("╔══════════════════════════════════════╗");
  console.log("║        🤖  Web RAG Chatbot           ║");
  console.log("╚══════════════════════════════════════╝");
  console.log('Type "exit" at any time to quit.\n');

  // ── Step 1: Get URL ──
  let url = "";
  while (!url.startsWith("http")) {
    url = (await prompt(rl, "Enter a URL to scrape: ")).trim();
    if (!url.startsWith("http")) {
      console.log("⚠️  Please enter a valid URL starting with http:// or https://");
    }
  }

  // ── Step 2: Init ChromaDB ──
  try {
    await initChroma();
  } catch (err) {
    console.error(`\n❌ Could not connect to ChromaDB: ${err.message}`);
    console.error("   Make sure your container is running: docker compose up -d");
    rl.close();
    return;
  }

  // ── Step 3: Ingest ──
  try {
    await ingest(url);
  } catch (err) {
    console.error(`\n❌ Failed to scrape: ${err.message}`);
    rl.close();
    return;
  }

  const docCount = await collection.count();
  if (docCount === 0) {
    console.log("❌ No content was extracted. Try a different URL.");
    rl.close();
    return;
  }

  console.log("💬 Ask anything about the page. Type \"exit\" to quit.\n");

  // ── Step 3: Chat loop ──
  while (true) {
    const userInput = (await prompt(rl, "You: ")).trim();

    if (!userInput) continue;
    if (userInput.toLowerCase() === "exit") {
      console.log("\n👋 Goodbye!");
      rl.close();
      break;
    }

    try {
      process.stdout.write("Bot: ");
      const answer = await chat(userInput);
      console.log(answer + "\n");
    } catch (err) {
      console.error(`\n❌ Error: ${err.message}\n`);
    }
  }
}

main();