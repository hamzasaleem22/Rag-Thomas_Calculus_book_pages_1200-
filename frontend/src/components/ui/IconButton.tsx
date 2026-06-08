import { type ReactNode } from "react";

interface IconButtonProps {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  tooltip?: string;
  className?: string;
  variant?: "default" | "danger";
}

const variantClasses: Record<NonNullable<IconButtonProps["variant"]>, string> = {
  default: "text-slate-500 hover:text-slate-700 hover:bg-slate-100",
  danger: "text-slate-500 hover:text-red-500 hover:bg-red-50",
};

export default function IconButton({
  children,
  onClick,
  disabled = false,
  tooltip,
  className = "",
  variant = "default",
}: IconButtonProps) {
  return (
    <div className="relative group inline-flex">
      <button
        type="button"
        onClick={onClick}
        disabled={disabled}
        className={`w-8 h-8 flex items-center justify-center p-2 rounded-xl transition-colors
          ${variantClasses[variant]}
          ${className}`}
      >
        {children}
      </button>
      {tooltip && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2.5 py-1
          bg-slate-800 text-white text-xs font-medium rounded-lg whitespace-nowrap
          opacity-0 invisible group-hover:opacity-100 group-hover:visible
          transition-all duration-200 pointer-events-none z-50
          shadow-lg shadow-slate-900/20">
          {tooltip}
          {/* Tooltip arrow */}
          <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-px
            border-4 border-transparent border-t-slate-800" />
        </div>
      )}
    </div>
  );
}
