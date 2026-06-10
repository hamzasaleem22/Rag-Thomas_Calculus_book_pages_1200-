/* ──────────────────────────────────────────────
   Shared TypeScript Interfaces
   Thomas' Calculus RAG — Frontend
   ────────────────────────────────────────────── */

/** A single citation from the RAG pipeline */
export interface Citation {
  text: string;
  page: number | null;
  chapter: string | null;
  section: string | null;
}

/** A chat message (user or assistant) */
export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  sources?: string[];
  timestamp: number;
}

/** A persisted chat session */
export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

/** A single dynamic section from the parsed answer */
export interface AnswerSection {
  header: string;
  content: string;
}

/** Parsed structure from the LLM answer text */
export interface ParsedAnswer {
  summary: string;
  keyPoints: string[];
  formulas: string[];
  sections: AnswerSection[];
  rawText: string;
}

/** A widget block type for rendering parsed answers */
export type WidgetType = "summary" | "keypoints" | "formula" | "sources";

export interface WidgetBlock {
  type: WidgetType;
  content: string | string[];
  citationRefs?: number[];
}

/** API request payload */
export interface QueryRequest {
  query: string;
  top_k: number;
  rerank: boolean;
  use_mmr?: boolean;
}

/** API response from /query (non-streaming) */
export interface QueryResponse {
  answer: string;
  citations: Citation[];
  sources: string[];
}

/** SSE event types from /query/stream */
export type StreamEventType = "token" | "citations" | "error";

export interface StreamTokenEvent {
  type: "token";
  content: string;
}

export interface StreamCitationsEvent {
  type: "citations";
  citations: Citation[];
}

export interface StreamErrorEvent {
  type: "error";
  content: string;
}

export type StreamEvent = StreamTokenEvent | StreamCitationsEvent | StreamErrorEvent;

/** Callbacks for streaming */
export interface StreamCallbacks {
  onToken: (token: string) => void;
  onCitations: (citations: Citation[]) => void;
  onDone: () => void;
  onError: (error: string) => void;
}

/** Chat state exposed by useChat hook */
export interface UseChatReturn {
  messages: Message[];
  isStreaming: boolean;
  sendMessage: (text: string) => void;
  regenerateMessage: (messageId: string) => void;
  deleteMessage: (messageId: string) => void;
  clearChat: () => void;
  cancelStream: () => void;
  loadMessages: (messages: Message[]) => void;
}

/** Chat history state exposed by useChatHistory hook */
export interface UseChatHistoryReturn {
  sessions: ChatSession[];
  activeSessionId: string | null;
  newSession: () => void;
  switchSession: (id: string) => void;
  deleteSession: (id: string) => void;
  saveCurrentSession: (messages: Message[]) => void;
}
