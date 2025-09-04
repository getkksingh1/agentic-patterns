# Agentic Patterns Website

The complete guide to agentic patterns - a comprehensive website covering 21 essential chapters for building autonomous AI agents, from foundational techniques to advanced scaling and safety considerations.

## 📖 Features

- **Complete Guide Structure**: 21 chapters organized in 4 progressive parts
- **Interactive Table of Contents**: Beautiful, navigable guide structure
- **Detailed Chapters**: Each with code examples, practical applications, and next steps
- **Comprehensive Appendices**: 7 additional resources and practical guides
- **Modern Design**: Built with Next.js 14, TypeScript, and Tailwind CSS
- **Responsive**: Optimized for all devices and screen sizes

## 📋 Content Structure

### Part One – Foundations of Agentic Patterns (7 Chapters)
1. Prompt Chaining
2. Routing
3. Parallelization
4. Reflection
5. Tool Use
6. Planning
7. Multi-Agent

### Part Two – Learning and Adaptation (4 Chapters)
8. Memory Management
9. Learning and Adaptation
10. Model Context Protocol (MCP)
11. Goal Setting and Monitoring

### Part Three – Human-Centric Patterns (3 Chapters)
12. Exception Handling and Recovery
13. Human-in-the-Loop
14. Knowledge Retrieval (RAG)

### Part Four – Scaling, Safety, and Discovery (7 Chapters)
15. Inter-Agent Communication (A2A)
16. Resource-Aware Optimization
17. Reasoning Techniques
18. Guardrails / Safety Patterns
19. Evaluation and Monitoring
20. Prioritization
21. Exploration and Discovery

### Appendices (7 Additional Resources)
- A: Advanced Prompting Techniques
- B: AI Agentic: From GUI to Real World Environment
- C: Quick Overview of Agentic Frameworks
- D: Building an Agent with AgentSpace (online only)
- E: AI Agents on the CLI (online only)
- F: Under the Hood: An Inside Look at the Agents' Reasoning Engines
- G: Coding Agents

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
```

2. Run the development server:
```bash
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser.

## 🏗️ Project Structure

```
src/
├── app/                       # Next.js 13+ app directory
│   ├── layout.tsx            # Root layout with metadata
│   ├── page.tsx              # Home page with hero and TOC
│   ├── globals.css           # Global styles
│   ├── table-of-contents/    # Full TOC page
│   ├── chapters/[slug]/      # Dynamic chapter pages
│   ├── appendix/[slug]/      # Dynamic appendix pages
│   ├── patterns/             # All chapters listing
│   └── about/                # About page
├── components/               # Reusable UI components
│   ├── Header.tsx           # Navigation header
│   ├── Footer.tsx           # Site footer
│   ├── Hero.tsx             # Landing page hero
│   ├── TableOfContents.tsx  # Interactive TOC
│   ├── ChapterContent.tsx   # Chapter page layout
│   ├── AppendixContent.tsx  # Appendix page layout
│   └── ...
└── data/                     # Content and data
    ├── chapters.ts          # Chapter definitions with content
    └── appendices.ts        # Appendix content
```

## 🛠️ Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

## 🎯 Key Features

- **Comprehensive Content**: 21 detailed chapters with practical code examples
- **Progressive Structure**: From basic foundations to advanced enterprise patterns
- **Interactive Navigation**: Easy browsing between chapters and sections
- **Code Examples**: Real-world implementations for each pattern
- **Practical Applications**: Use cases and implementation guidance
- **Modern Tech Stack**: Built with the latest web technologies

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Submit bug reports and feature requests
- Contribute new chapter content or improvements
- Enhance code examples and implementations
- Improve documentation and resources

## 📄 License

This project is open source and available under the MIT License.
