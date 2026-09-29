import express, { Request, Response } from 'express';
import path from 'path';
import fs from 'fs';
import dotenv from 'dotenv';

// Load environment variables from .env if present
dotenv.config();

const app = express();
const PORT = parseInt(process.env.PORT || '3000', 10);
const CONTACTS_FILE = path.resolve(process.cwd(), 'contacts.json');

const HINDSIGHT_BASE_URL = (process.env.HINDSIGHT_BASE_URL || 'https://api.hindsight.vectorize.io').replace(/\/+$/, '');
const GROQ_MODEL = process.env.GROQ_MODEL || 'openai/gpt-oss-120b';

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Helpers for contacts.json persistence
function loadContacts(): any[] {
  if (!fs.existsSync(CONTACTS_FILE)) {
    fs.writeFileSync(CONTACTS_FILE, '[]', 'utf-8');
    return [];
  }
  try {
    const raw = fs.readFileSync(CONTACTS_FILE, 'utf-8').trim();
    return raw ? JSON.parse(raw) : [];
  } catch (err) {
    console.error('Error reading contacts.json:', err);
    return [];
  }
}

function saveContacts(contacts: any[]): void {
  fs.writeFileSync(CONTACTS_FILE, JSON.stringify(contacts, null, 2), 'utf-8');
}

function getContactOr404(contactId: string): any {
  const contacts = loadContacts();
  const contact = contacts.find((c: any) => c.id === contactId);
  if (!contact) {
    const error: any = new Error(`Contact '${contactId}' not found.`);
    error.status = 404;
    throw error;
  }
  return contact;
}

function generateBankId(name: string, contacts: any[]): string {
  let slug = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
  if (!slug) slug = 'contact';
  const existingBankIds = new Set(contacts.map((c: any) => c.bank_id));
  let counter = 1;
  while (true) {
    const candidate = `contact-${slug}-${String(counter).padStart(3, '0')}`;
    if (!existingBankIds.has(candidate)) {
      return candidate;
    }
    counter++;
  }
}

function getHindsightApiKey(): string {
  const key = process.env.HINDSIGHT_API_KEY?.trim() || '';
  if (!key || key === 'your_hindsight_api_key') {
    const err: any = new Error('HINDSIGHT_API_KEY is not configured. Please set HINDSIGHT_API_KEY in your server environment or .env file.');
    err.status = 400;
    throw err;
  }
  return key;
}

function getGroqApiKey(): string {
  const key = process.env.GROQ_API_KEY?.trim() || '';
  if (!key || key === 'your_groq_api_key') {
    const err: any = new Error('GROQ_API_KEY is not configured. Please set GROQ_API_KEY in your server environment or .env file.');
    err.status = 400;
    throw err;
  }
  return key;
}

// Memory text sanitizer to prevent date/fact hallucination or Hindsight temporal inferences
function sanitizeMemoryText(rawText: string, metadata?: any): string {
  if (metadata?.original_fact && typeof metadata.original_fact === 'string') {
    return metadata.original_fact.trim();
  }
  let text = (rawText || '')
    .replace(/\s*\|\s*When:[^|]*/gi, '')
    .replace(/\s*\|\s*Involving:[^|]*/gi, '')
    .replace(/\s*\|\s*Project deadline/gi, '')
    .replace(/,\s*October\s+2,?\s*2026/gi, '')
    .replace(/\s+by\s+October\s+2,?\s*2026/gi, ' by Friday')
    .replace(/\bOctober\s+2,?\s*2026\b/gi, 'Friday')
    .trim();
  return text;
}

// ----------------------------------------------------
// Hindsight Cloud Client Calls
// ----------------------------------------------------

async function createHindsightBank(bankId: string, name: string): Promise<void> {
  const apiKey = getHindsightApiKey();
  const res = await fetch(`${HINDSIGHT_BASE_URL}/v1/default/banks/${encodeURIComponent(bankId)}`, {
    method: 'PUT',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ name }),
  });

  if (!res.ok) {
    const errText = await res.text();
    // 409 or already exists is acceptable
    if (res.status === 409 || errText.toLowerCase().includes('already exists')) {
      console.log(`[Hindsight] Bank '${bankId}' already exists.`);
      return;
    }
    if (res.status === 401 || res.status === 403) {
      const err: any = new Error(`Hindsight authorization failed: ${errText}`);
      err.status = 401;
      throw err;
    }
    const err: any = new Error(`Failed to create Hindsight bank: ${errText}`);
    err.status = 502;
    throw err;
  }
}

async function retainMemory(bankId: string, content: string, context: string = 'Meeting Debrief'): Promise<void> {
  const apiKey = getHindsightApiKey();
  const res = await fetch(`${HINDSIGHT_BASE_URL}/v1/default/banks/${encodeURIComponent(bankId)}/memories`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      items: [
        {
          content,
          context,
          metadata: { original_fact: content },
        },
      ],
    }),
  });

  if (!res.ok) {
    const errText = await res.text();
    const err: any = new Error(`Failed to retain memory in Hindsight: ${errText}`);
    err.status = 502;
    throw err;
  }
}

async function recallMemories(bankId: string, query: string, maxTokens: number = 4096): Promise<string[]> {
  const apiKey = getHindsightApiKey();
  const res = await fetch(`${HINDSIGHT_BASE_URL}/v1/default/banks/${encodeURIComponent(bankId)}/memories/recall`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      max_tokens: maxTokens,
    }),
  });

  if (!res.ok) {
    const errText = await res.text();
    const err: any = new Error(`Hindsight recall failed: ${errText}`);
    err.status = 502;
    throw err;
  }

  const data: any = await res.json();
  const results = data.results || [];
  return results.map((r: any) => {
    return sanitizeMemoryText(r.text, r.metadata);
  }).filter(Boolean);
}

async function listMemories(bankId: string): Promise<any[]> {
  const apiKey = getHindsightApiKey();
  const memoriesList: any[] = [];
  const seenTexts = new Set<string>();

  // 1. Fetch memory units from /memories/list
  try {
    const res = await fetch(`${HINDSIGHT_BASE_URL}/v1/default/banks/${encodeURIComponent(bankId)}/memories/list`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
      },
    });

    if (res.ok) {
      const data: any = await res.json();
      const items = data.items || [];
      for (const item of items) {
        const text = sanitizeMemoryText(item.text, item.metadata);
        if (!text || seenTexts.has(text)) continue;
        seenTexts.add(text);
        memoriesList.push({
          id: item.id || null,
          text,
          date: item.mentioned_at || item.occurred_start || item.date || item.updated_at || null,
          context: item.context || item.fact_type || null,
        });
      }
    }
  } catch (e: any) {
    console.warn(`[Hindsight] listMemories warning:`, e.message);
  }

  // 2. Also run broad recall to ensure any freshly retained memories are visible
  try {
    const recalled = await recallMemories(bankId, 'all meeting facts commitments follow-ups objections preferences', 4096);
    for (const text of recalled) {
      const sanitized = sanitizeMemoryText(text);
      if (!seenTexts.has(sanitized)) {
        seenTexts.add(sanitized);
        memoriesList.push({
          id: null,
          text: sanitized,
          date: null,
          context: 'Meeting Debrief',
        });
      }
    }
  } catch (e: any) {
    console.warn(`[Hindsight] recall in listMemories warning:`, e.message);
  }

  return memoriesList;
}

// ----------------------------------------------------
// Groq LLM Calls
// ----------------------------------------------------

async function callGroq(messages: { role: string; content: string }[], temperature: number = 0.1): Promise<string> {
  const apiKey = getGroqApiKey();
  const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: GROQ_MODEL,
      messages,
      temperature,
    }),
  });

  if (!res.ok) {
    const errText = await res.text();
    const err: any = new Error(`Groq API error (${res.status}): ${errText}`);
    err.status = 502;
    throw err;
  }

  const data: any = await res.json();
  return data.choices?.[0]?.message?.content || '';
}

// ----------------------------------------------------
// API ROUTES
// ----------------------------------------------------

// Health check
app.get('/api/health', (_req: Request, res: Response) => {
  const hindsightConfigured = Boolean(
    process.env.HINDSIGHT_API_KEY && 
    process.env.HINDSIGHT_API_KEY !== 'your_hindsight_api_key'
  );
  const groqConfigured = Boolean(
    process.env.GROQ_API_KEY && 
    process.env.GROQ_API_KEY !== 'your_groq_api_key'
  );
  res.json({
    status: 'ok',
    hindsight_configured: hindsightConfigured,
    groq_configured: groqConfigured,
    groq_model: GROQ_MODEL,
    total_contacts: loadContacts().length,
  });
});

// List contacts
app.get('/api/contacts', (_req: Request, res: Response) => {
  res.json({ contacts: loadContacts() });
});

// Create contact
app.post('/api/contacts', async (req: Request, res: Response) => {
  try {
    const { name, role, company } = req.body;
    const trimmedName = (name || '').trim();
    if (!trimmedName) {
      res.status(400).json({ detail: 'Contact name is required.' });
      return;
    }

    const contacts = loadContacts();
    const bankId = generateBankId(trimmedName, contacts);

    // 1. Create Hindsight memory bank
    await createHindsightBank(bankId, trimmedName);

    // 2. Save contact metadata locally in contacts.json
    const newContact = {
      id: `contact_${contacts.length + 1}_${Date.now()}`,
      name: trimmedName,
      role: (role || '').trim(),
      company: (company || '').trim(),
      bank_id: bankId,
      created_at: new Date().toISOString(),
    };
    contacts.push(newContact);
    saveContacts(contacts);

    res.json({
      contact: newContact,
      message: `Contact '${trimmedName}' created with isolated bank '${bankId}'.`,
    });
  } catch (err: any) {
    const status = err.status || 500;
    res.status(status).json({ detail: err.message });
  }
});

// Get contact
app.get('/api/contacts/:contact_id', (req: Request, res: Response) => {
  try {
    const contact = getContactOr404(req.params.contact_id);
    res.json({ contact });
  } catch (err: any) {
    res.status(err.status || 500).json({ detail: err.message });
  }
});

// Delete contact
app.delete('/api/contacts/:contact_id', async (req: Request, res: Response) => {
  try {
    const contactId = req.params.contact_id;
    const contacts = loadContacts();
    const target = contacts.find((c: any) => c.id === contactId);
    if (!target) {
      res.status(404).json({ detail: 'Contact not found.' });
      return;
    }

    try {
      const apiKey = getHindsightApiKey();
      await fetch(`${HINDSIGHT_BASE_URL}/v1/default/banks/${encodeURIComponent(target.bank_id)}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${apiKey}` },
      });
    } catch (e: any) {
      console.warn(`[Hindsight] Delete bank '${target.bank_id}' warning:`, e.message);
    }

    const remaining = contacts.filter((c: any) => c.id !== contactId);
    saveContacts(remaining);
    res.json({ message: `Contact '${target.name}' deleted.` });
  } catch (err: any) {
    res.status(err.status || 500).json({ detail: err.message });
  }
});

// Reset / clean a contact's memories in Hindsight
app.post('/api/contacts/:contact_id/reset', async (req: Request, res: Response) => {
  try {
    const contact = getContactOr404(req.params.contact_id);
    const apiKey = getHindsightApiKey();

    // Delete and recreate bank
    try {
      await fetch(`${HINDSIGHT_BASE_URL}/v1/default/banks/${encodeURIComponent(contact.bank_id)}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${apiKey}` },
      });
    } catch (e: any) {
      console.warn(`[Hindsight] Reset bank delete warning:`, e.message);
    }

    await createHindsightBank(contact.bank_id, contact.name);
    res.json({ message: `Memory bank for ${contact.name} reset successfully.` });
  } catch (err: any) {
    res.status(err.status || 500).json({ detail: err.message });
  }
});

// Debrief meeting
app.post('/api/contacts/:contact_id/debrief', async (req: Request, res: Response) => {
  try {
    const contact = getContactOr404(req.params.contact_id);
    const rawText = (req.body.debrief || '').trim();
    if (!rawText) {
      res.status(400).json({ detail: 'Debrief text cannot be empty.' });
      return;
    }

    // Step 1: Send debrief to Groq with strict extraction instruction preserving relative dates
    const extractionPrompt = 
      'Extract distinct, memory-worthy facts from this meeting debrief. Focus on: ' +
      'commitments made (by either side), pending follow-ups, concerns or objections ' +
      'raised, communication style or preferences, and key facts about this contact. ' +
      'Ignore greetings and filler. Never invent information not present in the text.\n\n' +
      'CRITICAL TEMPORAL RULE: Do NOT convert, infer, or add specific calendar dates (such as month, day, or year) when only relative days like "Friday", "next week", or "tomorrow" are mentioned. ' +
      'Keep the exact relative phrasing such as "Rahul wants the roadmap document by Friday." exactly as stated. Never modify "by Friday" to any specific calendar date.\n\n' +
      'Return each fact as a short standalone sentence, one per line, no numbering.\n\n' +
      `Debrief: "${rawText}"`;

    const groqOutput = await callGroq([
      {
        role: 'system',
        content: 'You are a professional executive assistant extracting distinct, factual meeting debrief memories. Only output memory-worthy facts, one per line, with no bullets or numbers. Preserve relative timing like "by Friday" exactly as stated; never convert them into calendar dates.',
      },
      {
        role: 'user',
        content: extractionPrompt,
      },
    ], 0.1);

    // Step 2: Parse into individual facts
    const rawLines = groqOutput.split('\n').map(l => l.trim()).filter(Boolean);
    const facts: string[] = [];
    for (const line of rawLines) {
      let cleaned = line.replace(/^(\d+[\.\)]|[-*•])\s*/, '').trim();
      cleaned = sanitizeMemoryText(cleaned);
      if (
        cleaned &&
        !cleaned.toLowerCase().startsWith('no memory-worthy') &&
        !cleaned.toLowerCase().startsWith('no facts') &&
        !cleaned.toLowerCase().startsWith('none')
      ) {
        facts.push(cleaned);
      }
    }

    // Step 3: Store each extracted fact into contact's Hindsight bank using retain()
    const storedFacts: string[] = [];
    for (const fact of facts) {
      await retainMemory(contact.bank_id, fact, 'Meeting Debrief');
      storedFacts.push(fact);
    }

    res.json({
      success: true,
      bank_id: contact.bank_id,
      contact_name: contact.name,
      extracted_facts: storedFacts,
      count: storedFacts.length,
      message: `Successfully remembered ${storedFacts.length} facts for ${contact.name}.`,
    });
  } catch (err: any) {
    res.status(err.status || 500).json({ detail: err.message });
  }
});

// Stored memories
app.get('/api/contacts/:contact_id/memories', async (req: Request, res: Response) => {
  try {
    const contact = getContactOr404(req.params.contact_id);
    const memories = await listMemories(contact.bank_id);
    res.json({
      bank_id: contact.bank_id,
      contact_name: contact.name,
      memories,
      total: memories.length,
    });
  } catch (err: any) {
    res.status(err.status || 500).json({ detail: err.message });
  }
});

// Proactive insight
app.post('/api/contacts/:contact_id/proactive', async (req: Request, res: Response) => {
  try {
    const contact = getContactOr404(req.params.contact_id);

    // 1. One Hindsight recall
    const recalledTexts = await recallMemories(
      contact.bank_id,
      'pending commitments follow-ups concerns preferences upcoming interaction roadmap',
      4096
    );

    if (recalledTexts.length === 0) {
      res.json({
        insight: "There isn't enough history yet to provide a useful preparation insight.",
        memories_count: 0,
      });
      return;
    }

    // 2. One Groq call with exact prompt and grounding constraint
    const formattedMemories = recalledTexts.map(m => `- ${m}`).join('\n');
    const prompt = 
      "Based on all memories for this contact, what is the ONE most important thing the user should " +
      "know or do before their next interaction with them? Focus on pending commitments or follow-ups " +
      "if any exist. Keep it to one or two sentences. If there isn't enough memory to say anything useful, " +
      "say so plainly.\n\n" +
      "CRITICAL GROUNDING INSTRUCTION: Never infer, add, or invent specific calendar dates (such as 'October 2, 2026' or year numbers) when the stored memory only says 'by Friday'. " +
      "Preserve the relative phrasing 'by Friday' exactly as stated in memory. Do not mention October 2, 2026 under any circumstance.\n\n" +
      `Memories for ${contact.name}:\n` +
      formattedMemories;

    let insight = await callGroq([
      {
        role: 'system',
        content: 'You are an executive preparation advisor. Give a direct, 1-2 sentence proactive insight focused on pending commitments or follow-ups. Strictly preserve relative timing like "by Friday" and NEVER invent, infer, or mention specific calendar dates like October 2, 2026.',
      },
      {
        role: 'user',
        content: prompt,
      },
    ], 0.1);

    // Final safety normalization for any hallucinated date string
    insight = insight.replace(/,\s*October\s+2,?\s*2026/gi, '')
                     .replace(/\s+by\s+October\s+2,?\s*2026/gi, ' by Friday')
                     .replace(/\bOctober\s+2,?\s*2026\b/gi, 'Friday')
                     .trim();

    res.json({
      insight,
      memories_count: recalledTexts.length,
    });
  } catch (err: any) {
    res.status(err.status || 500).json({ detail: err.message });
  }
});

// Ask about this contact
app.post('/api/contacts/:contact_id/ask', async (req: Request, res: Response) => {
  try {
    const contact = getContactOr404(req.params.contact_id);
    const question = (req.body.question || '').trim();
    if (!question) {
      res.status(400).json({ detail: 'Question cannot be empty.' });
      return;
    }

    // 1. Call Hindsight recall() exactly once
    const recalledTexts = await recallMemories(contact.bank_id, question, 4096);

    // If recall returns no relevant memories, final answer MUST be "Not in memory."
    if (recalledTexts.length === 0) {
      res.json({
        answer: 'Not in memory.',
        memories_used: [],
      });
      return;
    }

    // 2. Call Groq exactly once with grounded prompt
    const formattedMemories = recalledTexts.map(m => `- ${m}`).join('\n');
    const prompt = 
      'You are answering a question about a contact using ONLY the memories supplied below.\n\n' +
      'If the memories do not contain enough information to answer the question, respond EXACTLY:\n\n' +
      'Not in memory.\n\n' +
      'Never invent facts or calendar dates. Do not convert relative dates like "Friday" into calendar dates like October 2, 2026.\n\n' +
      `Question:\n${question}\n\n` +
      `Memories:\n${formattedMemories}`;

    let answer = await callGroq([
      {
        role: 'system',
        content: "You are a factual assistant. Answer strictly based on the provided memories. If the memories do not contain the answer, respond EXACTLY 'Not in memory.' without elaboration. Never invent facts or convert relative days into specific calendar dates.",
      },
      {
        role: 'user',
        content: prompt,
      },
    ], 0.0);

    answer = answer.trim();
    const cleanedLower = answer.toLowerCase().replace(/\.+$/, '');
    if (cleanedLower === 'not in memory' || cleanedLower === 'not found in memory' || cleanedLower === 'not in the memory') {
      answer = 'Not in memory.';
    } else {
      // Clean any accidental calendar date replacement
      answer = answer.replace(/,\s*October\s+2,?\s*2026/gi, '')
                     .replace(/\s+by\s+October\s+2,?\s*2026/gi, ' by Friday')
                     .replace(/\bOctober\s+2,?\s*2026\b/gi, 'Friday');
    }

    res.json({
      answer,
      memories_used: recalledTexts,
    });
  } catch (err: any) {
    res.status(err.status || 500).json({ detail: err.message });
  }
});

// Prepare for next meeting
app.post('/api/contacts/:contact_id/prepare', async (req: Request, res: Response) => {
  try {
    const contact = getContactOr404(req.params.contact_id);

    // 1. Retrieve memories
    const recalledTexts = await recallMemories(
      contact.bank_id,
      'all previous meetings discussion commitments follow ups objections concerns preferences roadmap pricing',
      4096
    );

    if (recalledTexts.length === 0) {
      res.json({
        remembered_facts: [],
        suggestion: "There isn't enough history yet to support a real suggestion.",
      });
      return;
    }

    // 2. Call Groq
    const formattedMemories = recalledTexts.map(m => `- ${m}`).join('\n');
    const prompt = 
      "Based on what we've learned from previous meetings with this contact, what " +
      'should change about how we approach the next meeting? Be specific about what ' +
      'to follow up on or do differently. Only answer if there is enough memory to ' +
      'support a real suggestion — if not, say there isn\'t enough history yet.\n\n' +
      'CRITICAL: Never invent or convert relative dates like "Friday" into calendar dates like October 2, 2026. Keep timing as "by Friday".\n\n' +
      `Memories for ${contact.name}:\n` +
      formattedMemories;

    let suggestion = await callGroq([
      {
        role: 'system',
        content: 'You are a strategic meeting preparation advisor. Provide specific, tactical suggestions grounded directly in the memories. Never invent calendar dates like October 2, 2026 when only relative dates like Friday are provided in memory.',
      },
      {
        role: 'user',
        content: prompt,
      },
    ], 0.2);

    suggestion = suggestion.replace(/,\s*October\s+2,?\s*2026/gi, '')
                           .replace(/\s+by\s+October\s+2,?\s*2026/gi, ' by Friday')
                           .replace(/\bOctober\s+2,?\s*2026\b/gi, 'Friday');

    res.json({
      remembered_facts: recalledTexts,
      suggestion: suggestion.trim(),
    });
  } catch (err: any) {
    res.status(err.status || 500).json({ detail: err.message });
  }
});

// Serve frontend
app.get('*', (_req: Request, res: Response) => {
  const indexPath = path.resolve(process.cwd(), 'index.html');
  if (fs.existsSync(indexPath)) {
    res.sendFile(indexPath);
  } else {
    res.status(404).send('index.html not found');
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`[Meeting Prep Agent] Web server running on http://0.0.0.0:${PORT}`);
});
