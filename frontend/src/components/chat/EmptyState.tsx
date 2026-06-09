import GlassCard from "../ui/GlassCard";

interface EmptyStateProps {
  onExampleClick: (text: string) => void;
}

interface ExamplePrompt {
  emoji: string;
  text: string;
  delay: string;
}

const examples: ExamplePrompt[] = [
  {
    emoji: "💡",
    text: "What is the chain rule and how does it work?",
    delay: "100ms",
  },
  {
    emoji: "📐",
    text: "Explain Taylor series with examples",
    delay: "200ms",
  },
  {
    emoji: "∫",
    text: "How to solve integrals using substitution?",
    delay: "300ms",
  },
];

export default function EmptyState({ onExampleClick }: EmptyStateProps) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center px-6 py-12">
      {/* ── Hero Icon ── */}
      <div className="animate-float mb-6">
        <span className="gradient-text text-5xl select-none" aria-hidden="true">
          ◆
        </span>
      </div>

      {/* ── Title ── */}
      <h1 className="gradient-text text-3xl font-extrabold tracking-tight text-center">
        Thomas' Calculus AI
      </h1>
      <p className="text-slate-500 text-base mt-2 text-center">
        Ask anything from the 14th Edition
      </p>

      {/* ── Example Prompt Cards ── */}
      <div className="flex flex-col gap-3 mt-10 w-full max-w-md">
        {examples.map((example) => (
          <div
            key={example.text}
            className="animate-fade-in-up"
            style={{ animationDelay: example.delay }}
          >
            <GlassCard
              hover
              glow
              onClick={() => onExampleClick(example.text)}
              className="p-4 w-full"
            >
              <div className="flex items-center gap-3.5">
                <span className="text-xl shrink-0 select-none" aria-hidden="true">
                  {example.emoji}
                </span>
                <span className="text-sm text-slate-700 leading-relaxed">
                  {example.text}
                </span>
              </div>
            </GlassCard>
          </div>
        ))}
      </div>
    </div>
  );
}
