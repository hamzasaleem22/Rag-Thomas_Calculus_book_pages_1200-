import { type ReactNode } from "react";

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  hover?: boolean;
  strong?: boolean;
  onClick?: () => void;
}

export default function GlassCard({
  children,
  className = "",
  hover = false,
  strong = false,
  onClick,
}: GlassCardProps) {
  return (
    <div
      onClick={onClick}
      className={`${strong ? "glass-strong" : "glass"} ${hover ? "glass-hover" : ""} rounded-2xl ${onClick ? "cursor-pointer" : ""} ${className}`}
    >
      {children}
    </div>
  );
}
