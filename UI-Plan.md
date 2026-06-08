
# UI-Plan.md — Thomas' Calculus RAG Frontend Upgrade

> **Goal:** Transform the current single-file dark-theme chat UI into a modern, professional, animated **light-theme** interface inspired by Upwork/glassmorphism aesthetics — with structured answer widgets, LaTeX math rendering, streaming responses, chat history persistence, and rich micro-interactions.

---

## 1. Design System & Visual Theme

### 1.1 Theme Philosophy: Glassmorphism + Gradients
A **white/light-gray base** with vibrant **blue-to-teal gradient** accents, frosted-glass (glassmorphism) cards, soft shadows, and rounded geometry. The feel should be energetic yet professional — like Upwork meets Linear.

### 1.2 Color Palette

| Role | Value | Tailwind Class |
|------|-------|----------------|
| **Background** | `#F8FAFC` | `bg-slate-50` |
| **Surface (cards)** | `rgba(255,255,255,0.7)` + `backdrop-blur-xl` | glass effect |
| **Surface Solid** | `#FFFFFF` | `bg-white` |
| **Border** | `rgba(148,163,184,0.2)` | `border-slate-200/20` |
| **Text Primary** | `#0F172A` | `text-slate-900` |
| **Text Secondary** | `#475569` | `text-slate-600` |
| **Text Muted** | `#94A3B8` | `text-slate-400` |
| **Gradient Primary** | `#3B82F6` → `#06B6D4` | `bg-gradient-to-r from-blue-500 to-cyan-500` |
| **Gradient Accent** | `#8B5CF6` → `#3B82F6` | `bg-gradient-to-r from-violet-500 to-blue-500` |
| **Gradient Soft** | `#EFF6FF` → `#ECFEFF` | `bg-gradient-to-br from-blue-50 to-cyan-50` |
| **User Bubble** | Gradient primary (blue→cyan) | `bg-gradient-to-r from-blue-500 to-cyan-500 text-white` |
| **Success/Correct** | `#10B981` | `text-emerald-500` |
| **Error** | `#EF4444` | `text-red-500` |

### 1.3 Typography

| Level | Font | Size | Weight | Tracking |
|-------|------|------|--------|----------|
| **Display (Hero)** | `Inter` (system fallback) | `2rem` | `800` | `-0.025em` |
| **Heading** | Inter | `1.25rem` | `700` | `-0.02em` |
| **Body** | Inter | `0.938rem` | `400` | `0` |
| **Small/Caption** | Inter | `0.813rem` | `500` | `0.01em` |
| **Code/Math** | `JetBrains Mono` / KaTeX fonts | `0.875rem` | `400` | `0` |

**Font Loading:** Add Google Fonts `Inter` (400,500,600,700,800) and `JetBrains Mono` (400) via `<link>` in `index.html`.

### 1.4 Shape & Elevation

| Element | Border Radius | Shadow |
|---------|--------------|--------|
| **Cards / Bubbles** | `1rem` (rounded-2xl) | `shadow-sm` default, `shadow-md` on hover |
| **Buttons** | `0.75rem` (rounded-xl) | `shadow-sm` |
| **Input** | `1rem` (rounded-2xl) | `shadow-sm`, `shadow-lg` on focus |
| **Glass Panels** | `1.5rem` (rounded-3xl) | `shadow-lg shadow-blue-500/5` |
| **Citation Chips** | `0.5rem` (rounded-lg) | none |

---

## 2. Architecture: Component Breakdown

The monolithic `App.tsx` (141 lines) will be decomposed into a proper component tree:

```
frontend/src/
├── App.tsx                          ← Root layout + state orchestration
├── main.tsx                         ← Entry point (unchanged)
├── index.css                        ← Tailwind + custom animations + glass utilities
│
├── types/
│   └── index.ts                     ← Shared TypeScript interfaces
│
├── hooks/
│   ├── useChat.ts                   ← Chat state, send, stream, localStorage persistence
│   └── useChatHistory.ts            ← Chat session management (list, switch, delete, new)
│
├── services/
│   └── api.ts                       ← API client (query, streamQuery, health check)
│
├── utils/
│   ├── answerParser.ts              ← Parse LLM answer into structured widgets (Summary, Key Points, Formulas)
│   └── storage.ts                   ← localStorage read/write helpers with JSON serialization
│
├── components/
│   ├── layout/
│   │   ├── Header.tsx               ← Top bar: logo, title, refresh button, history toggle
│   │   └── ChatContainer.tsx        ← Scrollable message area with auto-scroll
│   │
│   ├── chat/
│   │   ├── MessageBubble.tsx        ← Routes to UserBubble or AssistantBubble
│   │   ├── UserBubble.tsx           ← User message: gradient pill, right-aligned
│   │   ├── AssistantBubble.tsx      ← Bot answer: glass card with structured widget children
│   │   ├── EmptyState.tsx           ← Hero-style empty state with example prompts
│   │   ├── StreamingIndicator.tsx   ← Animated typing/streaming cursor
│   │   └── MessageActions.tsx       ← Copy, regenerate, delete actions (hover toolbar)
│   │
│   ├── widgets/
│   │   ├── AnswerSummary.tsx        ← Summary card widget (main answer text)
│   │   ├── KeyPointsList.tsx        ← Bullet-pointed key insights widget
│   │   ├── FormulaBox.tsx           ← Highlighted math/formula widget with LaTeX render
│   │   ├── CitationCard.tsx         ← Individual citation with page/chapter badge
│   │   └── SourcesWidget.tsx       ← Collapsible citations container with chips
│   │
│   ├── input/
│   │   └── ChatInput.tsx            ← Input form with animated send button, keyboard shortcuts
│   │
│   └── ui/
│       ├── GradientButton.tsx       ← Reusable gradient button component
│       ├── GlassCard.tsx            ← Reusable glassmorphism card wrapper
│       ├── IconButton.tsx           ← Small icon-only button (copy, refresh, etc.)
│       └── Badge.tsx                ← Small pill badge (page numbers, chapter labels)
```

---

## 3. Feature Specifications

### 3.1 Structured Answer Widgets (Bot Responses as Widget Cards)

The backend LLM already structures answers with sections like **Answer**, **Key Points**, **Formula**. The `answerParser.ts` utility will parse the response text and split it into structured blocks:

```
┌─────────────────────────────────────────────────┐
│  🤖  Assistant                                   │
│                                                   │
│  ┌─────────────────────────────────────────────┐ │
│  │  📝 SUMMARY                                 │ │
│  │  "The derivative of sin(x) is cos(x).       │ │
│  │   This is one of the fundamental... [1]"    │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  ┌─────────────────────────────────────────────┐ │
│  │  🔑 KEY POINTS                              │ │
│  │  • d/dx[sin(x)] = cos(x)                    │ │
│  │  • Applies to all real x                    │ │
│  │  • Derived from limit definition [2]        │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  ┌─────────────────────────────────────────────┐ │
│  │  ∫ FORMULA                                  │ │
│  │  ┌──────────────────────────┐               │ │
│  │  │  d/dx [sin(x)] = cos(x)  │  ← KaTeX     │ │
│  │  └──────────────────────────┘               │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  ┌─────────────────────────────────────────────┐ │
│  │  📚 SOURCES (5)              [▼ Collapse]   │ │
│  │  [p.142 · Ch.3]  [p.145 · Ch.3]  [+3 more] │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│     [📋 Copy]  [🔄 Regenerate]  [🗑 Delete]     │
└─────────────────────────────────────────────────┘
```

**Parser Logic (`answerParser.ts`):**
- Detect sections by markdown-style headers: `**Answer:**`, `**Key Points:**`, `**Formula:**`, `### Formula`
- Extract citation markers `[N]` and map to citation objects
- Return `{ summary: string, keyPoints: string[], formulas: string[], citationRefs: number[] }`

### 3.2 LaTeX / KaTeX Math Rendering

- Install `react-katex` + `katex` packages
- Install KaTeX CSS via `@import` in `index.css`
- Inline math: `$...$` → render with `<InlineMath>`
- Display math: `$$...$$` → render with `<BlockMath>` inside FormulaBox widget
- Fallback: If KaTeX fails to parse, display raw LaTeX in `<code>` block
- FormulaBox widget gets a special glass card with a subtle gradient border

### 3.3 Streaming Response (SSE)

- Switch from `POST /api/query` to `POST /api/query/stream`
- `api.ts` creates an `EventSource`-like reader using `fetch()` + `ReadableStream`
- Tokens arrive as `data: {"type": "token", "content": "..."}` events
- Final event: `data: {"type": "citations", "citations": [...]}`
- `data: [DONE]` closes the stream
- UI shows a **blinking cursor** after the latest token during streaming
- Streaming text is appended to the current assistant message in real-time
- Citations appear only after stream completes (fade-in animation)
- **Cancel button** appears during streaming to abort the request

### 3.4 Chat History (LocalStorage Persistence)

**Data Model:**
```typescript
interface ChatSession {
  id: string;               // UUID
  title: string;            // Auto-generated from first user message (first 50 chars)
  messages: Message[];      // Full conversation
  createdAt: number;        // Timestamp
  updatedAt: number;        // Last message timestamp
}
```

**Behavior:**
- On app load: restore the most recent session from `localStorage`
- **Refresh/New Chat button** in header: saves current session, starts a blank one
- Auto-save: messages are persisted to `localStorage` after every send/receive
- Max stored sessions: **20** (oldest auto-pruned)
- Session list accessible via a **history dropdown** from the header (simple popover, not a sidebar)
- Each session shows: title, date, message count
- Click to switch sessions (current is saved, selected is loaded)
- Delete button per session in the dropdown

### 3.5 Message Actions (Hover Toolbar)

Each assistant message reveals a floating action toolbar on hover:

| Action | Icon | Behavior |
|--------|------|----------|
| **Copy** | Clipboard icon | Copies answer text to clipboard, shows "Copied!" tooltip for 2s |
| **Regenerate** | Refresh icon | Re-sends the preceding user message as a new query |
| **Delete** | Trash icon | Removes this user+assistant message pair with fade-out animation |

User messages show: **Copy** and **Delete** only.

### 3.6 Empty State / Welcome Screen

When no messages exist, display a centered hero section:

```
┌─────────────────────────────────────────┐
│                                          │
│      [Animated Gradient Logo/Icon]       │
│                                          │
│     Thomas' Calculus AI Assistant        │
│     ─────────────────────────────        │
│     Ask anything from the 14th Edition   │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │  💡 "What is the chain rule?"    │   │
│  └──────────────────────────────────┘   │
│  ┌──────────────────────────────────┐   │
│  │  📐 "Explain Taylor series"      │   │
│  └──────────────────────────────────┘   │
│  ┌──────────────────────────────────┐   │
│  │  ∫ "How to integrate by parts?"  │   │
│  └──────────────────────────────────┘   │
│                                          │
└─────────────────────────────────────────┘
```

- Example prompts are clickable → auto-fills the input
- Subtle floating animation on the cards
- Gradient text on the title

---

## 4. Animations & Micro-Interactions

### 4.1 Global Animation Tokens (in `index.css`)

```css
/* Custom keyframes */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes slideInRight {
  from { opacity: 0; transform: translateX(20px); }
  to { opacity: 1; transform: translateX(0); }
}

@keyframes slideInLeft {
  from { opacity: 0; transform: translateX(-20px); }
  to { opacity: 1; transform: translateX(0); }
}

@keyframes pulseGlow {
  0%, 100% { box-shadow: 0 0 0 0 rgba(59,130,246,0.3); }
  50% { box-shadow: 0 0 20px 4px rgba(59,130,246,0.15); }
}

@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}

@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-6px); }
}

@keyframes gradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
```

### 4.2 Animation Map

| Element | Animation | Duration | Easing |
|---------|-----------|----------|--------|
| **New user message** | `slideInRight` + `fadeIn` | 300ms | `ease-out` |
| **New assistant message** | `slideInLeft` + `fadeIn` | 400ms | `ease-out` |
| **Widget cards inside answer** | Staggered `fadeInUp` (100ms delay each) | 350ms | `ease-out` |
| **Citations expand** | `fadeInUp` + height transition | 300ms | `ease-in-out` |
| **Empty state cards** | `float` (infinite, 3s cycle) | — | `ease-in-out` |
| **Header gradient** | `gradientShift` (subtle background animation) | 8s | `linear` infinite |
| **Send button hover** | Scale `1.05` + `pulseGlow` | 200ms | `ease-out` |
| **Loading indicator** | Shimmer gradient bar (not "Thinking..." text) | 1.5s | `linear` infinite |
| **Streaming cursor** | `blink` | 800ms | `step-end` infinite |
| **Message delete** | `fadeOut` + `scale(0.95)` + height collapse | 250ms | `ease-in` |
| **Hover toolbar** | `fadeIn` + `translateY(-4px)` | 150ms | `ease-out` |
| **Page load** | Entire app `fadeIn` | 500ms | `ease-out` |
| **Session switch** | Chat area `fadeIn` | 300ms | `ease-out` |

### 4.3 No External Animation Library
All animations implemented with **pure CSS** + Tailwind's `@keyframes` and `animation` utilities. No need for `framer-motion` — keeps the bundle lean.

---

## 5. New Dependencies to Install

```bash
npm install react-katex katex
npm install -D @types/react-katex
```

| Package | Purpose | Size |
|---------|---------|------|
| `react-katex` | React wrapper for KaTeX math rendering | ~5KB |
| `katex` | Fast math typesetting engine | ~200KB (with fonts) |
| `@types/react-katex` | TypeScript definitions | dev only |

> **Note:** No animation library, no component library, no router. Keeping dependencies minimal.

---

## 6. Implementation Phases

### Phase 1: Foundation & Design System
**Priority: Critical | Files: `index.css`, `index.html`, `types/index.ts`, `utils/storage.ts`**

1. Update `index.html` — add Google Fonts (Inter, JetBrains Mono), fix title to "Thomas' Calculus AI", add meta description
2. Rewrite `index.css` — Tailwind import + custom keyframes + glass utility classes + custom scrollbar styles
3. Create `types/index.ts` — all shared interfaces (`Message`, `Citation`, `ChatSession`, `ParsedAnswer`, `WidgetBlock`)
4. Create `utils/storage.ts` — localStorage CRUD helpers with session management
5. Delete dead `App.css` file

### Phase 2: Core UI Components
**Priority: Critical | Files: All `components/` files**

1. Build `GlassCard.tsx` — reusable frosted-glass container with hover shadow
2. Build `GradientButton.tsx` — gradient button with hover scale + glow
3. Build `IconButton.tsx` — small icon button with tooltip
4. Build `Badge.tsx` — pill badge for page/chapter labels
5. Build `Header.tsx` — glass header with gradient title, refresh button, history dropdown trigger
6. Build `ChatInput.tsx` — glass input bar with animated send button, disabled states
7. Build `EmptyState.tsx` — hero welcome screen with floating example prompts

### Phase 3: Chat Engine & Streaming
**Priority: Critical | Files: `services/api.ts`, `hooks/useChat.ts`, `hooks/useChatHistory.ts`**

1. Create `services/api.ts`:
   - `queryStream(params, onToken, onCitations, onDone, onError)` — SSE fetch using `ReadableStream`
   - `querySync(params)` — fallback non-streaming query
   - `healthCheck()` — ping `/health`
2. Create `hooks/useChat.ts`:
   - State: `messages`, `isStreaming`, `currentSessionId`
   - Actions: `sendMessage`, `regenerateMessage`, `deleteMessage`, `clearChat`
   - Auto-persist to localStorage on every state change
   - Restore last session on mount
3. Create `hooks/useChatHistory.ts`:
   - State: `sessions[]`, `activeSessionId`
   - Actions: `newSession`, `switchSession`, `deleteSession`, `getSessions`
   - Auto-prune sessions beyond 20

### Phase 4: Answer Widgets & LaTeX
**Priority: High | Files: `utils/answerParser.ts`, All `widgets/` files**

1. Create `utils/answerParser.ts`:
   - Parse LLM response text into `{ summary, keyPoints[], formulas[], rawText }`
   - Detect section headers: `**Answer:**`, `**Key Points:**`, `### Formula`, etc.
   - Extract `[N]` citation references and map to citation objects
2. Build `AnswerSummary.tsx` — glass card with the main answer paragraph, inline math via KaTeX
3. Build `KeyPointsList.tsx` — styled bullet list inside a glass card with numbered items
4. Build `FormulaBox.tsx` — special glass card with gradient border, renders LaTeX via `BlockMath`
5. Build `CitationCard.tsx` — compact card showing page badge, chapter, truncated text
6. Build `SourcesWidget.tsx` — collapsible container holding CitationCards with chip count

### Phase 5: Message Bubbles & Actions
**Priority: High | Files: `components/chat/` files**

1. Build `UserBubble.tsx` — right-aligned gradient pill with slide-in animation
2. Build `AssistantBubble.tsx` — left-aligned glass card containing widget children
   - Staggered fadeInUp animation for each widget
   - Shows streaming cursor during SSE
   - Renders parsed widgets or falls back to plain text
3. Build `MessageBubble.tsx` — router component that renders User or Assistant bubble
4. Build `MessageActions.tsx` — hover toolbar (copy, regenerate, delete) with tooltip feedback
5. Build `StreamingIndicator.tsx` — animated shimmer bar + blinking cursor (replaces "Thinking...")

### Phase 6: Assembly & Polish
**Priority: High | Files: `App.tsx`, `ChatContainer.tsx`**

1. Build `ChatContainer.tsx` — scrollable area with auto-scroll, empty state routing
2. Rewrite `App.tsx` — compose all components, wire hooks, handle state orchestration
3. Add custom scrollbar styling (thin, themed to match glass aesthetic)
4. Responsive: ensure mobile layout works (full-width bubbles, stacked widgets)
5. Add keyboard shortcuts: `Enter` to send, `Shift+Enter` for newline
6. Final polish: test all animations, verify streaming works, confirm localStorage persistence

---

## 7. File Change Summary

### New Files (22 files)
| File | Lines (est.) | Purpose |
|------|-------------|---------|
| `src/types/index.ts` | ~60 | Shared TypeScript interfaces |
| `src/utils/storage.ts` | ~80 | localStorage helpers |
| `src/utils/answerParser.ts` | ~120 | Parse LLM answers into widgets |
| `src/services/api.ts` | ~100 | API client (streaming + sync) |
| `src/hooks/useChat.ts` | ~120 | Chat state management |
| `src/hooks/useChatHistory.ts` | ~80 | Session list management |
| `src/components/layout/Header.tsx` | ~80 | Top navigation bar |
| `src/components/layout/ChatContainer.tsx` | ~50 | Scrollable chat area |
| `src/components/chat/MessageBubble.tsx` | ~20 | Message router |
| `src/components/chat/UserBubble.tsx` | ~30 | User message bubble |
| `src/components/chat/AssistantBubble.tsx` | ~80 | Bot answer with widgets |
| `src/components/chat/EmptyState.tsx` | ~70 | Welcome screen |
| `src/components/chat/StreamingIndicator.tsx` | ~30 | Streaming animation |
| `src/components/chat/MessageActions.tsx` | ~60 | Hover action toolbar |
| `src/components/widgets/AnswerSummary.tsx` | ~30 | Summary card widget |
| `src/components/widgets/KeyPointsList.tsx` | ~40 | Key points widget |
| `src/components/widgets/FormulaBox.tsx` | ~40 | LaTeX formula widget |
| `src/components/widgets/CitationCard.tsx` | ~35 | Citation display card |
| `src/components/widgets/SourcesWidget.tsx` | ~50 | Collapsible sources |
| `src/components/input/ChatInput.tsx` | ~60 | Input form |
| `src/components/ui/GlassCard.tsx` | ~25 | Glass card wrapper |
| `src/components/ui/GradientButton.tsx` | ~25 | Gradient button |
| `src/components/ui/IconButton.tsx` | ~30 | Icon-only button |
| `src/components/ui/Badge.tsx` | ~15 | Pill badge |

### Modified Files (3 files)
| File | Change |
|------|--------|
| `src/App.tsx` | Complete rewrite — compose new component tree, wire hooks |
| `src/index.css` | Rewrite — Tailwind + custom animations + glass utilities + scrollbar |
| `index.html` | Add fonts, fix title, add meta tags |

### Deleted Files (1 file)
| File | Reason |
|------|--------|
| `src/App.css` | Dead code — never imported, Vite scaffold leftover |

---

## 8. Responsive Design Strategy

| Breakpoint | Layout |
|------------|--------|
| **Mobile (< 640px)** | Full-width bubbles, stacked widgets, compact header, bottom input bar |
| **Tablet (640-1024px)** | Max-width chat `42rem`, centered, 2-column citation grid |
| **Desktop (> 1024px)** | Max-width chat `48rem`, centered, 3-column citation grid, history dropdown as popover |

---

## 9. Performance Targets

| Metric | Target |
|--------|--------|
| **Bundle size** (gzipped) | < 300KB (including KaTeX) |
| **First Contentful Paint** | < 1.5s |
| **Time to Interactive** | < 2.5s |
| **Lighthouse Performance** | > 90 |
| **Animation FPS** | 60fps (CSS-only animations, no JS animation loops) |

---

## 10. Quick Reference: What Changes vs Current State

| Aspect | Current | After Upgrade |
|--------|---------|---------------|
| **Theme** | Dark (neutral-900) | Light glassmorphism + gradients |
| **Files** | 1 monolithic App.tsx | 25+ modular components |
| **Answer Display** | Plain text `whitespace-pre-wrap` | Structured widget cards (Summary, Key Points, Formula, Sources) |
| **Math** | Raw LaTeX text | Beautiful KaTeX rendering |
| **API Mode** | Synchronous POST | SSE streaming with real-time tokens |
| **Persistence** | None (lost on refresh) | localStorage with 20 session history |
| **Refresh/New Chat** | Manual browser refresh | Header button — saves session, starts new |
| **Message Actions** | None | Copy, Regenerate, Delete per message |
| **Animations** | None | 12+ CSS animations (slide, fade, shimmer, float, glow) |
| **Empty State** | Plain text | Interactive hero with clickable example prompts |
| **Loading** | "Thinking..." text | Animated shimmer bar + streaming cursor |
| **Citations** | `<details>` collapse | Glass card chips with page/chapter badges |
| **Input** | Basic input + button | Glass input bar with animated gradient send button |
| **Typography** | System default | Inter + JetBrains Mono + KaTeX fonts |
| **Page Title** | "frontend" | "Thomas' Calculus AI" |
