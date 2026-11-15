// The client wires and the one upstream wire this baseline speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This baseline forwards both to a
// provider on the openai wire and translates back:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to the provider and streams
//     the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// The gemini and anthropic upstream wires are not implemented in this baseline (main.ts answers 501 for a
// provider on either): a real entry needs their request and event shapes too, which is the biggest single
// gap in this starter.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/* ------------------------------------------------------------------ */
/* Added for the aggregator: cross-wire upstream bodies, answer       */
/* parsing, and client answer builders.                               */
/* ------------------------------------------------------------------ */

export interface ParsedAnswer {
  text: string;
  promptTokens: number | null;
  completionTokens: number | null;
  stop: string | null;
}

const FIXED_CREATED = 1767225600;

function cloneJson<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function stringOrEmpty(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function messageText(value: unknown): string {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    return value
      .filter((b: any) => isRecord(b) && b.type === "text" && typeof b.text === "string")
      .map((b: any) => b.text)
      .join("\n");
  }
  return "";
}

/* ---------------- OpenAI upstream body ---------------- */

export function openAIUpstreamBody(body: any, model: string): any {
  const out = cloneJson(body);
  out.model = model;
  if (out.stream === true) {
    out.stream_options = { include_usage: true };
  }
  return out;
}

/* ---------------- Anthropic upstream body ---------------- */

function openAIContentToAnthropicBlocks(content: any): any[] {
  if (typeof content === "string") return [{ type: "text", text: content }];
  if (!Array.isArray(content)) return [];
  const blocks: any[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === "text" && typeof part.text === "string") {
      blocks.push({ type: "text", text: part.text });
    } else if (part.type === "image_url" && isRecord(part.image_url) && typeof part.image_url.url === "string") {
      const url = part.image_url.url;
      const m = /^data:([^;]+);base64,(.*)$/.exec(url);
      if (m) {
        blocks.push({ type: "image", source: { type: "base64", media_type: m[1], data: m[2] } });
      } else {
        blocks.push({ type: "image", source: { type: "url", url } });
      }
    } else if (part.type === "input_audio") {
      // No audio in the sim; drop.
    }
  }
  return blocks;
}

function openAIToolChoiceToAnthropic(choice: any): any {
  if (choice === "auto") return { type: "auto" };
  if (choice === "required" || choice === "any") return { type: "any" };
  if (isRecord(choice) && choice.type === "function" && isRecord(choice.function) && typeof choice.function.name === "string") {
    return { type: "tool", name: choice.function.name };
  }
  return { type: "auto" };
}

export function anthropicUpstreamBody(body: any, model: string, clientApi: "openai" | "anthropic"): any {
  if (clientApi === "anthropic") {
    // Already an Anthropic body: pass through, model replaced.
    const out = cloneJson(body);
    out.model = model;
    return out;
  }
  const out: any = { model };
  const systemTexts: string[] = [];
  const messages: any[] = [];
  for (const msg of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(msg)) continue;
    if (msg.role === "system") {
      const t = messageText(msg.content);
      if (t !== "") systemTexts.push(t);
      continue;
    }
    if (msg.role === "tool") {
      messages.push({
        role: "user",
        content: [{ type: "tool_result", tool_use_id: typeof msg.tool_call_id === "string" ? msg.tool_call_id : "", content: messageText(msg.content) }],
      });
      continue;
    }
    if (msg.role === "assistant") {
      const blocks: any[] = openAIContentToAnthropicBlocks(msg.content);
      if (Array.isArray(msg.tool_calls)) {
        for (const tc of msg.tool_calls) {
          if (isRecord(tc) && isRecord(tc.function) && typeof tc.function.name === "string") {
            let input: any = {};
            try {
              input = typeof tc.function.arguments === "string" ? JSON.parse(tc.function.arguments) : tc.function.arguments ?? {};
            } catch {
              input = {};
            }
            blocks.push({ type: "tool_use", id: typeof tc.id === "string" ? tc.id : "", name: tc.function.name, input });
          }
        }
      }
      messages.push({ role: "assistant", content: blocks.length > 0 ? blocks : "" });
      continue;
    }
    // user
    messages.push({ role: "user", content: openAIContentToAnthropicBlocks(msg.content) });
  }
  if (systemTexts.length > 0) out.system = systemTexts.join("\n");
  out.messages = messages;
  if (Array.isArray(body.tools)) {
    out.tools = body.tools
      .filter((t: any) => isRecord(t) && isRecord(t.function) && typeof t.function.name === "string")
      .map((t: any) => ({
        name: t.function.name,
        description: typeof t.function.description === "string" ? t.function.description : undefined,
        input_schema: isRecord(t.function.parameters) ? t.function.parameters : { type: "object" },
      }));
  }
  if (body.tool_choice !== undefined) out.tool_choice = openAIToolChoiceToAnthropic(body.tool_choice);
  const maxTokens = typeof body.max_tokens === "number" ? body.max_tokens : typeof body.max_completion_tokens === "number" ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (typeof body.top_p === "number") out.top_p = body.top_p;
  if (body.stop !== undefined) out.stop_sequences = typeof body.stop === "string" ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s: any) => typeof s === "string") : undefined;
  out.stream = body.stream === true;
  return out;
}

/* ---------------- Gemini upstream body ---------------- */

function openAIMessageToGeminiParts(msg: any): any[] {
  const parts: any[] = [];
  const content = msg.content;
  if (typeof content === "string") {
    if (content !== "") parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === "text" && typeof part.text === "string") {
      parts.push({ text: part.text });
    } else if (part.type === "image_url" && isRecord(part.image_url) && typeof part.image_url.url === "string") {
      const url = part.image_url.url;
      const m = /^data:([^;]+);base64,(.*)$/.exec(url);
      if (m) parts.push({ inlineData: { mimeType: m[1], data: m[2] } });
      else parts.push({ fileData: { fileUri: url } });
    }
  }
  return parts;
}

function anthropicContentToGeminiParts(content: any): any[] {
  const parts: any[] = [];
  if (typeof content === "string") {
    if (content !== "") parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === "text" && typeof block.text === "string") {
      parts.push({ text: block.text });
    } else if (block.type === "image" && isRecord(block.source)) {
      if (block.source.type === "base64" && typeof block.source.data === "string") {
        parts.push({ inlineData: { mimeType: typeof block.source.media_type === "string" ? block.source.media_type : "image/png", data: block.source.data } });
      } else if (block.source.type === "url" && typeof block.source.url === "string") {
        parts.push({ fileData: { fileUri: block.source.url } });
      }
    }
  }
  return parts;
}

export function geminiUpstreamBody(body: any, model: string, clientApi: "openai" | "anthropic"): any {
  const isAnthropic = clientApi === "anthropic";
  const out: any = {};
  const systemParts: any[] = [];
  const contents: any[] = [];
  const declarations: any[] = [];
  if (isAnthropic) {
    if (typeof body.system === "string") {
      if (body.system !== "") systemParts.push({ text: body.system });
    } else if (Array.isArray(body.system)) {
      for (const block of body.system) {
        if (isRecord(block) && block.type === "text" && typeof block.text === "string") systemParts.push({ text: block.text });
      }
    }
    for (const msg of body.messages) {
      const parts = anthropicContentToGeminiParts(msg.content);
      if (msg.role === "assistant") {
        if (parts.length > 0) contents.push({ role: "model", parts });
      } else {
        contents.push({ role: "user", parts });
      }
    }
    if (Array.isArray(body.tools)) {
      for (const t of body.tools) {
        if (isRecord(t) && typeof t.name === "string") {
          declarations.push({ name: t.name, description: typeof t.description === "string" ? t.description : undefined, parameters: isRecord(t.input_schema) ? t.input_schema : undefined });
        }
      }
    }
    const gen: any = {};
    if (typeof body.max_tokens === "number") gen.maxOutputTokens = body.max_tokens;
    if (typeof body.temperature === "number") gen.temperature = body.temperature;
    if (Array.isArray(body.stop_sequences)) gen.stopSequences = body.stop_sequences.filter((s: any) => typeof s === "string");
    if (Object.keys(gen).length > 0) out.generationConfig = gen;
  } else {
    for (const msg of Array.isArray(body.messages) ? body.messages : []) {
      if (!isRecord(msg)) continue;
      if (msg.role === "system") {
        for (const p of openAIMessageToGeminiParts(msg)) systemParts.push(p);
        continue;
      }
      if (msg.role === "tool") continue;
      const parts = openAIMessageToGeminiParts(msg);
      if (msg.role === "assistant") {
        if (parts.length > 0) contents.push({ role: "model", parts });
      } else {
        contents.push({ role: "user", parts });
      }
    }
    if (Array.isArray(body.tools)) {
      for (const t of body.tools) {
        if (isRecord(t) && isRecord(t.function) && typeof t.function.name === "string") {
          declarations.push({ name: t.function.name, description: typeof t.function.description === "string" ? t.function.description : undefined, parameters: isRecord(t.function.parameters) ? t.function.parameters : undefined });
        }
      }
    }
    const gen: any = {};
    const maxTokens = typeof body.max_tokens === "number" ? body.max_tokens : typeof body.max_completion_tokens === "number" ? body.max_completion_tokens : undefined;
    if (maxTokens !== undefined) gen.maxOutputTokens = maxTokens;
    if (typeof body.temperature === "number") gen.temperature = body.temperature;
    if (body.stop !== undefined) gen.stopSequences = typeof body.stop === "string" ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s: any) => typeof s === "string") : undefined;
    if (isRecord(body.response_format) && (body.response_format.type === "json_object" || body.response_format.type === "json_schema")) {
      gen.responseMimeType = "application/json";
    }
    if (Object.keys(gen).length > 0) out.generationConfig = gen;
  }
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  out.contents = contents;
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  return out;
}

/* ---------------- SSE frame splitting (shared) ---------------- */

export function splitSseFrames(text: string): { event: string; data: string }[] {
  const frames: { event: string; data: string }[] = [];
  const chunks = text.split(/\r?\n\r?\n/);
  for (const chunk of chunks) {
    if (chunk.trim() === "") continue;
    let event = "";
    const dataLines: string[] = [];
    for (const line of chunk.split(/\r?\n/)) {
      if (line.startsWith("event:")) event = line.slice("event:".length).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice("data:".length).replace(/^ /, ""));
    }
    if (dataLines.length === 0) continue;
    frames.push({ event, data: dataLines.join("\n") });
  }
  return frames;
}

/* ---------------- Answer parsing (per upstream wire) ---------------- */

export function parseOpenAIResponse(text: string, stream: boolean): ParsedAnswer | null {
  try {
    if (!stream) {
      const j = JSON.parse(text);
      if (!isRecord(j) || !Array.isArray(j.choices) || j.choices.length === 0) return null;
      const choice = j.choices[0];
      if (!isRecord(choice) || !isRecord(choice.message)) return null;
      const content = typeof choice.message.content === "string" ? choice.message.content : "";
      const usage = isRecord(j.usage) ? j.usage : null;
      return {
        text: content,
        promptTokens: usage && typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : null,
        completionTokens: usage && typeof usage.completion_tokens === "number" ? usage.completion_tokens : null,
        stop: typeof choice.finish_reason === "string" ? choice.finish_reason : null,
      };
    }
    let out = "";
    let prompt: number | null = null;
    let completion: number | null = null;
    let stop: string | null = null;
    let sawDone = false;
    for (const frame of splitSseFrames(text)) {
      const payload = frame.data.trim();
      if (payload === "[DONE]") {
        sawDone = true;
        continue;
      }
      if (sawDone) continue;
      let chunk: any;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk) || !Array.isArray(chunk.choices)) continue;
      if (isRecord(chunk.usage)) {
        if (typeof chunk.usage.prompt_tokens === "number") prompt = chunk.usage.prompt_tokens;
        if (typeof chunk.usage.completion_tokens === "number") completion = chunk.usage.completion_tokens;
      }
      if (chunk.choices.length === 0) continue;
      const choice = chunk.choices[0];
      if (!isRecord(choice)) continue;
      if (isRecord(choice.delta) && typeof choice.delta.content === "string") out += choice.delta.content;
      if (typeof choice.finish_reason === "string") stop = choice.finish_reason;
    }
    return { text: out, promptTokens: prompt, completionTokens: completion, stop };
  } catch {
    return null;
  }
}

export function parseAnthropicResponse(text: string, stream: boolean): ParsedAnswer | null {
  try {
    if (!stream) {
      const j = JSON.parse(text);
      if (!isRecord(j) || j.type !== "message" || !Array.isArray(j.content)) return null;
      let out = "";
      for (const block of j.content) {
        if (isRecord(block) && block.type === "text" && typeof block.text === "string") out += block.text;
      }
      const usage = isRecord(j.usage) ? j.usage : null;
      return {
        text: out,
        promptTokens: usage && typeof usage.input_tokens === "number" ? usage.input_tokens : null,
        completionTokens: usage && typeof usage.output_tokens === "number" ? usage.output_tokens : null,
        stop: typeof j.stop_reason === "string" ? j.stop_reason : null,
      };
    }
    let out = "";
    let input: number | null = null;
    let output: number | null = null;
    let stop: string | null = null;
    for (const frame of splitSseFrames(text)) {
      let ev: any;
      try {
        ev = JSON.parse(frame.data);
      } catch {
        continue;
      }
      if (!isRecord(ev)) continue;
      const type = frame.event || ev.type;
      if (type === "message_start" && isRecord(ev.message) && isRecord(ev.message.usage)) {
        if (typeof ev.message.usage.input_tokens === "number") input = ev.message.usage.input_tokens;
        if (typeof ev.message.usage.output_tokens === "number") output = ev.message.usage.output_tokens;
      } else if (type === "content_block_delta" && isRecord(ev.delta) && ev.delta.type === "text_delta" && typeof ev.delta.text === "string") {
        out += ev.delta.text;
      } else if (type === "message_delta") {
        if (isRecord(ev.delta) && typeof ev.delta.stop_reason === "string") stop = ev.delta.stop_reason;
        if (isRecord(ev.usage) && typeof ev.usage.output_tokens === "number") output = ev.usage.output_tokens;
      }
    }
    return { text: out, promptTokens: input, completionTokens: output, stop };
  } catch {
    return null;
  }
}

function geminiPartsText(parts: any): string {
  if (!Array.isArray(parts)) return "";
  let out = "";
  for (const p of parts) {
    if (isRecord(p) && typeof p.text === "string") out += p.text;
  }
  return out;
}

export function parseGeminiResponse(text: string, stream: boolean): ParsedAnswer | null {
  try {
    if (!stream) {
      const j = JSON.parse(text);
      if (!isRecord(j) || !Array.isArray(j.candidates) || j.candidates.length === 0) return null;
      const cand = j.candidates[0];
      const content = isRecord(cand) && isRecord(cand.content) ? cand.content : null;
      const usage = isRecord(j.usageMetadata) ? j.usageMetadata : null;
      return {
        text: content ? geminiPartsText(content.parts) : "",
        promptTokens: usage && typeof usage.promptTokenCount === "number" ? usage.promptTokenCount : null,
        completionTokens: usage && typeof usage.candidatesTokenCount === "number" ? usage.candidatesTokenCount : null,
        stop: isRecord(cand) && typeof cand.finishReason === "string" ? cand.finishReason : null,
      };
    }
    let out = "";
    let prompt: number | null = null;
    let completion: number | null = null;
    let stop: string | null = null;
    for (const frame of splitSseFrames(text)) {
      let ev: any;
      try {
        ev = JSON.parse(frame.data);
      } catch {
        continue;
      }
      if (!isRecord(ev)) continue;
      if (Array.isArray(ev.candidates)) {
        for (const cand of ev.candidates) {
          if (isRecord(cand) && isRecord(cand.content)) out += geminiPartsText(cand.content.parts);
          if (isRecord(cand) && typeof cand.finishReason === "string") stop = cand.finishReason;
        }
      }
      if (isRecord(ev.usageMetadata)) {
        if (typeof ev.usageMetadata.promptTokenCount === "number") prompt = ev.usageMetadata.promptTokenCount;
        if (typeof ev.usageMetadata.candidatesTokenCount === "number") completion = ev.usageMetadata.candidatesTokenCount;
      }
    }
    return { text: out, promptTokens: prompt, completionTokens: completion, stop };
  } catch {
    return null;
  }
}

/* ---------------- Client answer builders ---------------- */

export function openAICompletion(model: string, ans: ParsedAnswer): any {
  const p = ans.promptTokens ?? 0;
  const c = ans.completionTokens ?? 0;
  return {
    id: "chatcmpl-agg",
    object: "chat.completion",
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: "assistant", content: ans.text }, finish_reason: "stop", logprobs: null }],
    usage: { prompt_tokens: p, completion_tokens: c, total_tokens: p + c },
  };
}

export function openAIStreamText(model: string, ans: ParsedAnswer): string {
  const p = ans.promptTokens ?? 0;
  const c = ans.completionTokens ?? 0;
  const frames: any[] = [
    { id: "chatcmpl-agg", object: "chat.completion.chunk", created: FIXED_CREATED, model, choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }] },
  ];
  if (ans.text !== "") {
    frames.push({ id: "chatcmpl-agg", object: "chat.completion.chunk", created: FIXED_CREATED, model, choices: [{ index: 0, delta: { content: ans.text }, finish_reason: null }] });
  }
  frames.push({ id: "chatcmpl-agg", object: "chat.completion.chunk", created: FIXED_CREATED, model, choices: [{ index: 0, delta: {}, finish_reason: "stop" }] });
  frames.push({ id: "chatcmpl-agg", object: "chat.completion.chunk", created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: p, completion_tokens: c, total_tokens: p + c } });
  return frames.map((f) => `data: ${JSON.stringify(f)}\n\n`).join("") + "data: [DONE]\n\n";
}

export function anthropicMessage(model: string, ans: ParsedAnswer): any {
  return {
    id: "msg-agg",
    type: "message",
    role: "assistant",
    model,
    content: [{ type: "text", text: ans.text }],
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: { input_tokens: ans.promptTokens ?? 0, output_tokens: ans.completionTokens ?? 0 },
  };
}

export function anthropicStreamText(model: string, ans: ParsedAnswer): string {
  const p = ans.promptTokens ?? 0;
  const c = ans.completionTokens ?? 0;
  const events: [string, any][] = [
    ["message_start", { type: "message_start", message: { id: "msg-agg", type: "message", role: "assistant", model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: p, output_tokens: 0 } } }],
    ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }],
  ];
  if (ans.text !== "") {
    events.push(["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: ans.text } }]);
  }
  events.push(["content_block_stop", { type: "content_block_stop", index: 0 }]);
  events.push(["message_delta", { type: "message_delta", delta: { stop_reason: "end_turn", stop_sequence: null }, usage: { output_tokens: c } }]);
  events.push(["message_stop", { type: "message_stop" }]);
  return events.map(([event, payload]) => `event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`).join("");
}

/* ---------------- Client error bodies ---------------- */

export function openAIErrorBody(status: number, code: string, message: string): any {
  return { error: { message, type: "api_error", code }, status };
}

export function anthropicErrorBody(status: number, code: string, message: string): any {
  return { type: "error", error: { type: "api_error", message }, status };
}

