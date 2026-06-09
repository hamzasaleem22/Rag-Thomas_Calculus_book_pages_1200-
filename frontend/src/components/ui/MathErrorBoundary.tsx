import { Component, type ReactNode, type ErrorInfo } from "react";

interface MathErrorBoundaryProps {
  latex: string;
  children: ReactNode;
}

interface MathErrorBoundaryState {
  hasError: boolean;
}

export class MathErrorBoundary extends Component<
  MathErrorBoundaryProps,
  MathErrorBoundaryState
> {
  constructor(props: MathErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): MathErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(_error: Error, _info: ErrorInfo) {
    if (this.props.latex) {
      // Silently handled — incomplete LaTeX during streaming is expected
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <code className="text-sm font-mono text-slate-500 block whitespace-pre-wrap">
          {this.props.latex}
        </code>
      );
    }
    return this.props.children;
  }
}

export default MathErrorBoundary;
