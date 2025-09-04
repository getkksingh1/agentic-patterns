# Chapter Files Organization

This directory contains individual chapter files for better maintainability and organization.

## File Structure

Each chapter is stored in a separate TypeScript file following the naming convention:
- `chapter-XX-chapter-name.ts` where XX is the chapter number (zero-padded)

## Current Chapters

### Part One – Foundations of Agentic Patterns
- [x] `chapter-01-prompt-chaining.ts` - Prompt Chaining (Complete - 25 min read)
- [x] `chapter-02-routing.ts` - Routing (Complete - 20 min read)  
- [x] `chapter-03-parallelization.ts` - Parallelization (Complete - 18 min read)
- [x] `chapter-04-reflection.ts` - Reflection (Complete - 22 min read)
- [x] `chapter-05-tool-use.ts` - Tool Use (Complete - 15 min read)
- [x] `chapter-06-planning.ts` - Planning (Complete - 26 min read)
- [x] `chapter-07-multi-agent.ts` - Multi-Agent Collaboration (Complete - 30 min read)

### Part Two – Learning and Adaptation
- [x] `chapter-08-memory-management.ts` - Memory Management (Complete - 32 min read)
- [x] `chapter-09-learning-adaptation.ts` - Learning and Adaptation (Complete - 35 min read)
- [x] `chapter-10-model-context-protocol.ts` - Model Context Protocol (Complete - 28 min read)
- [x] `chapter-11-goal-setting-monitoring.ts` - Goal Setting and Monitoring (Complete - 32 min read)

### Part Three – Human-Centric Patterns
- [x] `chapter-12-exception-handling.ts` - Exception Handling and Recovery (Complete - 30 min read)
- [x] `chapter-13-human-in-loop.ts` - Human-in-the-Loop (Complete - 28 min read)
- [x] `chapter-14-knowledge-retrieval.ts` - Knowledge Retrieval (RAG) (Complete - 31 min read)

### Part Four – Scaling, Safety, and Discovery
- [x] `chapter-15-inter-agent-communication.ts` - Inter-Agent Communication (A2A) (Complete - 29 min read)
- [x] `chapter-16-resource-aware-optimization.ts` - Resource-Aware Optimization (Complete - 33 min read)
- [x] `chapter-17-reasoning-techniques.ts` - Reasoning Techniques (Complete - 31 min read)
- [x] `chapter-18-guardrails-safety-patterns.ts` - Guardrails / Safety Patterns (Complete - 28 min read)
- [x] `chapter-19-evaluation-monitoring.ts` - Evaluation and Monitoring (Complete - 32 min read)
- [ ] `chapter-20-prioritization.ts` - Prioritization
- [ ] `chapter-21-exploration-discovery.ts` - Exploration and Discovery

## Adding New Chapters

1. Create a new file following the naming convention
2. Import the `Chapter` type from `../types`
3. Export a chapter object with all required fields
4. Add the import to `index.ts`
5. Add the chapter to the main `chapters.ts` array
6. Update this README with the completion status

## Chapter Structure

Each chapter should follow this TypeScript structure:

```typescript
import { Chapter } from '../types'

export const chapterName: Chapter = {
  id: 'chapter-slug',
  number: 1,
  title: 'Chapter Title',
  part: 'Part Name',
  description: 'Brief description',
  readingTime: 'X min read',
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced',
  content: {
    overview: 'Detailed overview...',
    keyPoints: ['Point 1', 'Point 2', ...],
    codeExample: 'Code example...',
    practicalApplications: ['Application 1', ...],
    nextSteps: ['Step 1', ...]
  },
  sections: [/* Optional detailed sections */],
  practicalExamples: [/* Optional examples */],
  references: [/* Optional references */],
  navigation: {
    previous: { id: 'prev-id', title: 'Previous Title' },
    next: { id: 'next-id', title: 'Next Title' }
  }
}
```

## Benefits of This Structure

- **Maintainability**: Each chapter is self-contained and easy to edit
- **Scalability**: Adding new chapters doesn't make any single file too large
- **Team Collaboration**: Multiple people can work on different chapters simultaneously
- **Version Control**: Changes to individual chapters are isolated
- **Type Safety**: All chapters use the same TypeScript interface
