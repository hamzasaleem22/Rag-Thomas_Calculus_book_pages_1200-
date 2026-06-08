import { type ReactNode } from "react";

interface GradientButtonProps {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  className?: string;
  type?: "submit" | "button";
  size?: "sm" | "md" | "lg";
}

const sizeClasses: Record<NonNullable<GradientButtonProps["size"]>, string> = {
  sm: "px-3 py-1.5 text-xs",
  md: "px-5 py-2.5 text-sm",
  lg: "px-6 py-3 text-base",
};

export default function GradientButton({
  children,
  onClick,
  disabled = false,
  className = "",
  type = "button",
  size = "md",
}: GradientButtonProps) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`gradient-bg text-white font-medium rounded-xl
        hover:shadow-lg hover:shadow-blue-500/25 hover:scale-[1.03]
        disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:hover:shadow-none
        transition-all duration-200
        ${sizeClasses[size]}
        ${className}`}
    >
      {children}
    </button>
  );
}
