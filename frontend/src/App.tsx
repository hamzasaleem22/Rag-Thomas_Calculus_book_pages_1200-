import { useState, useRef, useEffect } from "react";

interface Citation {
  text: string;
  page: number | null;
  chapter: string | null;
  section: string | null;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  sources?: string[];
}

function App() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    const userMsg: Message = { role: "user", content: query };
    setMessages((prev) => [...prev, userMsg]);
    setQuery("");
    setLoading(true);

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 10, rerank: true }),
      });
      const data = await res.json();

      const assistantMsg: Message = {
        role: "assistant",
        content: data.answer,
        citations: data.citations,
        sources: data.sources,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Error contacting server." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-dvh bg-neutral-900 text-neutral-100">
      <header className="border-b border-neutral-700 p-4">
        <h1 className="text-xl font-bold">Thomas' Calculus RAG</h1>
        <p className="text-sm text-neutral-400">Ask questions about the 14th Edition</p>
      </header>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-neutral-500 mt-20">
            <p className="text-lg">Ask a question about Thomas' Calculus</p>
            <p className="text-sm mt-2">e.g. "What is the formula for the derivative of sin(x)?"</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                msg.role === "user"
                  ? "bg-blue-600 text-white"
                  : "bg-neutral-800 text-neutral-100"
              }`}
            >
              <div className="whitespace-pre-wrap">{msg.content}</div>
              {msg.citations && msg.citations.length > 0 && (
                <details className="mt-2 text-xs text-neutral-400">
                  <summary className="cursor-pointer hover:text-neutral-300">
                    Sources ({msg.citations.length})
                  </summary>
                  <div className="mt-1 space-y-1">
                    {msg.citations.map((c, j) => (
                      <div key={j} className="border-l-2 border-neutral-600 pl-2 py-1">
                        <span className="text-neutral-500">
                          p.{c.page}
                          {c.chapter && ` · ${c.chapter}`}
                        </span>
                        <p className="truncate">{c.text.slice(0, 150)}...</p>
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-neutral-800 rounded-2xl px-4 py-3 text-neutral-400">
              Thinking...
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={handleSubmit} className="border-t border-neutral-700 p-4">
        <div className="flex gap-2">
          <input
            className="flex-1 rounded-xl bg-neutral-800 px-4 py-3 text-sm outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Ask a question..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="rounded-xl bg-blue-600 px-5 py-3 text-sm font-medium disabled:opacity-50 hover:bg-blue-700"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}

export default App;
