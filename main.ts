// Free-model aggregator entry (free-model-aggregator-01).
//
// One server, two client wires (OpenAI, Anthropic), ten simulated
// free-tier providers on three upstream wires (OpenAI, Anthropic,
// Gemini). Routing: rating-based, budget-spread, adaptive failover.
//
// Structure follows the kit starter: readBrief -> probe -> buildCatalog
// (the one catalog object, printed by --list and POSTed to /register),
// listen on 0.0.0.0 at an ephemeral port, poll /work every 100 ms and
// exit 0 when it says done. Everything deterministic: no clock, no
// unseeded randomness.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  openAIError,
  openAIModelList,
  type ServedModel,
} from './lib/wire.ts';
import {
  openAIUpstreamBody,
  anthropicUpstreamBody,
  anthropicToOpenAI,
  geminiUpstreamBody,
  parseOpenAIResponse,
  parseAnthropicResponse,
  parseGeminiResponse,
  openAICompletion,
  openAIStreamText,
  anthropicMessage,
  anthropicStreamText,
  type ParsedAnswer,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Soft per-provider call quotas: the per-trial minima of the budget
 * ranges, so a provider is never planned past a call count every trial
 * can afford. Rating order still decides within the quota. */
const SOFT_QUOTA: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
/** Most upstream attempts for one client request. */
const MAX_ATTEMPTS = 6;
/** How long one upstream call may take before it counts as a refusal. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** Pause before reusing the same provider after it asked retry-after. */
const RETRY_AFTER_SLEEP_MS = 1000;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

interface ProviderState {
  calls: number;
  consecutiveRefusals: number;
  blacklisted: boolean;
  coolRequest: number;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

const providerState = new Map<string, ProviderState>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function stateFor(providerId: string): ProviderState {
  let st = providerState.get(providerId);
  if (st === undefined) {
    st = { calls: 0, consecutiveRefusals: 0, blacklisted: false, coolRequest: -1 };
    providerState.set(providerId, st);
  }
  return st;
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

function isJsonModeRequest(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  const rf = body.response_format;
  return isRecord(rf) && (rf.type === 'json_object' || rf.type === 'json_schema');
}

/**
 * Every free model that meets one request's requirements, best rating
 * first (ties: provider id, then model id — deterministic).
 * An Anthropic-wire provider cannot carry an OpenAI JSON-mode request:
 * the Anthropic wire has no response_format, so the echo would lose
 * json_mode and the request would fail its checks.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', body: Json): Candidate[] {
  const jsonMode = isJsonModeRequest(clientApi, body);
  const list: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      list.push({ probe, model, rating: row !== null ? row.rating_milli : 0 });
    }
  }
  list.sort(
    (a, b) =>
      b.rating - a.rating ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return list;
}

/** The best candidate for one request: highest rating among providers under their soft quota when any is, else the best available. Skips blacklisted providers, the just-refused one, and a once-refused provider for the next request. */
function selectCandidate(
  reqIndex: number,
  requires: Requirement,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  excludeProvider: string | null,
): Candidate | null {
  const usable = candidatesFor(requires, clientApi, body).filter((c) => {
    const st = stateFor(c.probe.provider.id);
    if (st.blacklisted) return false;
    if (excludeProvider !== null && c.probe.provider.id === excludeProvider) return false;
    if (reqIndex === st.coolRequest) return false;
    return true;
  });
  if (usable.length === 0) return null;
  const under = usable.filter((c) => stateFor(c.probe.provider.id).calls < (SOFT_QUOTA[c.probe.provider.id] ?? 0));
  return under.length > 0 ? under[0] : usable[0];
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

interface UpstreamResult {
  ok: boolean;
  status: number;
  retryAfter: boolean;
  text: string;
}

async function callUpstream(cand: Candidate, body: Json, stream: boolean, clientApi: 'openai' | 'anthropic'): Promise<UpstreamResult> {
  const probe = cand.probe;
  const base = joinBase(ENV_URL, probe.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(probe.provider.auth, probe.provider.key),
  };
  let url: string;
  let reqBody: unknown;
  if (probe.provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    // An Anthropic client body must be rewritten to Chat Completions
    // (system -> system message, tools, stops); passing it through
    // verbatim loses system/tools/stops on the OpenAI parse.
    reqBody = clientApi === 'anthropic' ? anthropicToOpenAI(body, cand.model.id) : openAIUpstreamBody(body, cand.model.id);
  } else if (probe.provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    reqBody = anthropicUpstreamBody(body, cand.model.id, clientApi);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(cand.model.id)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
    reqBody = geminiUpstreamBody(body, cand.model.id, clientApi);
    if (stream) headers.accept = 'text/event-stream';
  }
  const resp = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(reqBody),
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
  });
  const text = await resp.text();
  const retryAfter = resp.headers.get('retry-after') !== null || /"retry_after"\s*:/.test(text);
  return { ok: resp.ok, status: resp.status, retryAfter, text };
}

function parseUpstream(wire: 'openai' | 'anthropic' | 'gemini', text: string, stream: boolean): ParsedAnswer | null {
  if (wire === 'openai') return parseOpenAIResponse(text, stream);
  if (wire === 'anthropic') return parseAnthropicResponse(text, stream);
  return parseGeminiResponse(text, stream);
}

/**
 * Serve one client request: pick the best free model for it, call it,
 * and on a 402/429/5xx retry on the next candidate. A 2xx is used as it
 * stands — never retried, so the trial's one_provider check holds.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const requires = requirementsFor(body);
  const reqIndex = manifestIndex - 1;
  const stream = body.stream === true;
  let excluded: string | null = null;
  let lastRetryAfter = false;

  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    let cand = selectCandidate(reqIndex, requires, clientApi, body, excluded);
    if (cand === null && excluded !== null) {
      // Only the just-refused provider remains usable: reuse it, pausing
      // first if it asked for a retry-after.
      if (lastRetryAfter) await sleep(RETRY_AFTER_SLEEP_MS);
      cand = selectCandidate(reqIndex, requires, clientApi, body, null);
    }
    if (cand === null) break;
    const providerId = cand.probe.provider.id;
    const st = stateFor(providerId);

    let result: UpstreamResult;
    try {
      result = await callUpstream(cand, body, stream, clientApi);
    } catch {
      // Network error / timeout: the provider did not answer. Treat as a
      // refusal for blacklist/cooling, but it consumed no budget.
      st.consecutiveRefusals += 1;
      if (st.consecutiveRefusals >= 2) st.blacklisted = true;
      st.coolRequest = reqIndex + 1;
      excluded = providerId;
      lastRetryAfter = false;
      continue;
    }
    st.calls += 1;
    if (!result.ok) {
      st.consecutiveRefusals += 1;
      if (st.consecutiveRefusals >= 2) st.blacklisted = true;
      st.coolRequest = reqIndex + 1;
      excluded = providerId;
      lastRetryAfter = result.retryAfter;
      continue;
    }

    st.consecutiveRefusals = 0;
    const ans = parseUpstream(cand.probe.provider.wire, result.text, stream);
    if (ans === null || !ans.text.startsWith('ANSWER ECHO')) {
      // Cannot happen in the sim; fail the request rather than retry
      // after a 2xx.
      return sendError(res, clientApi, 502, 'bad_gateway', 'upstream answer could not be parsed');
    }

    if (cand.probe.provider.wire === clientApi) {
      // Same wire: forward the provider answer verbatim. Byte-exact
      // passthrough is the safest wire fidelity there is.
      res.writeHead(200, {
        'content-type': stream ? 'text/event-stream' : 'application/json',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
      });
      res.end(result.text);
      return;
    }
    if (clientApi === 'openai') {
      if (stream) {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        res.end(openAIStreamText(cand.model.id, ans));
      } else {
        sendJson(res, 200, openAICompletion(cand.model.id, ans));
      }
    } else {
      if (stream) {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        res.end(anthropicStreamText(cand.model.id, ans));
      } else {
        sendJson(res, 200, anthropicMessage(cand.model.id, ans));
      }
    }
    return;
  }
  sendError(res, clientApi, 503, 'overloaded_error', 'no provider could serve this request');
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
