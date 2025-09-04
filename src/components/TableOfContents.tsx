'use client'

import Link from 'next/link'
import { Book, Code, Users, Shield, FileText, ChevronRight } from 'lucide-react'

interface Chapter {
  id: string
  number: number
  title: string
  hasCode?: boolean
}

interface Part {
  id: string
  title: string
  description: string
  icon: typeof Book
  color: string
  chapters: Chapter[]
}

interface AppendixItem {
  id: string
  title: string
  onlineOnly?: boolean
}

const parts: Part[] = [
  {
    id: 'foundations',
    title: 'Part One – Foundations of Agentic Patterns',
    description: 'Essential building blocks for autonomous AI agents',
    icon: Book,
    color: 'green',
    chapters: [
      { id: 'prompt-chaining', number: 1, title: 'Prompt Chaining', hasCode: true },
      { id: 'routing', number: 2, title: 'Routing', hasCode: true },
      { id: 'parallelization', number: 3, title: 'Parallelization', hasCode: true },
      { id: 'reflection', number: 4, title: 'Reflection', hasCode: true },
      { id: 'tool-use', number: 5, title: 'Tool Use', hasCode: true },
      { id: 'planning', number: 6, title: 'Planning', hasCode: true },
      { id: 'multi-agent', number: 7, title: 'Multi-Agent', hasCode: true },
    ]
  },
  {
    id: 'learning',
    title: 'Part Two – Learning and Adaptation',
    description: 'Memory, learning, and adaptive behaviors',
    icon: Code,
    color: 'blue',
    chapters: [
      { id: 'memory-management', number: 8, title: 'Memory Management', hasCode: true },
      { id: 'learning-adaptation', number: 9, title: 'Learning and Adaptation', hasCode: true },
      { id: 'model-context-protocol', number: 10, title: 'Model Context Protocol (MCP)', hasCode: true },
      { id: 'goal-setting-monitoring', number: 11, title: 'Goal Setting and Monitoring', hasCode: true },
    ]
  },
  {
    id: 'human-centric',
    title: 'Part Three – Human-Centric Patterns',
    description: 'Human-AI collaboration and interaction patterns',
    icon: Users,
    color: 'purple',
    chapters: [
      { id: 'exception-handling', number: 12, title: 'Exception Handling and Recovery', hasCode: true },
      { id: 'human-in-loop', number: 13, title: 'Human-in-the-Loop', hasCode: true },
      { id: 'knowledge-retrieval', number: 14, title: 'Knowledge Retrieval (RAG)', hasCode: true },
    ]
  },
  {
    id: 'scaling-safety',
    title: 'Part Four – Scaling, Safety, and Discovery',
    description: 'Enterprise-grade patterns for production systems',
    icon: Shield,
    color: 'red',
    chapters: [
      { id: 'inter-agent-communication', number: 15, title: 'Inter-Agent Communication (A2A)', hasCode: true },
      { id: 'resource-aware-optimization', number: 16, title: 'Resource-Aware Optimization', hasCode: true },
      { id: 'reasoning-techniques', number: 17, title: 'Reasoning Techniques', hasCode: true },
      { id: 'guardrails-safety', number: 18, title: 'Guardrails / Safety Patterns', hasCode: true },
      { id: 'evaluation-monitoring', number: 19, title: 'Evaluation and Monitoring', hasCode: true },
      { id: 'prioritization', number: 20, title: 'Prioritization', hasCode: true },
      { id: 'exploration-discovery', number: 21, title: 'Exploration and Discovery', hasCode: true },
    ]
  }
]

const appendices: AppendixItem[] = [
  { id: 'advanced-prompting-techniques', title: 'Advanced Prompting Techniques' },
  { id: 'ai-agentic-interactions', title: 'AI Agentic: From GUI to Real World Environment' },
  { id: 'agentic-frameworks-overview', title: 'Quick Overview of Agentic Frameworks' },
  { id: 'building-agent-agentspace', title: 'Building an Agent with AgentSpace', onlineOnly: true },
  { id: 'ai-agents-cli', title: 'AI Agents on the CLI', onlineOnly: true },
  { id: 'under-the-hood', title: 'Under the Hood: An Inside Look at the Agents\' Reasoning Engines' },
  { id: 'coding-agents', title: 'Coding Agents' },
]

export default function TableOfContents() {
  const getColorClasses = (color: string) => {
    const colors = {
      green: {
        bg: 'bg-green-50',
        border: 'border-green-200',
        icon: 'bg-green-100 text-green-600',
        title: 'text-green-800'
      },
      blue: {
        bg: 'bg-blue-50',
        border: 'border-blue-200',
        icon: 'bg-blue-100 text-blue-600',
        title: 'text-blue-800'
      },
      purple: {
        bg: 'bg-purple-50',
        border: 'border-purple-200',
        icon: 'bg-purple-100 text-purple-600',
        title: 'text-purple-800'
      },
      red: {
        bg: 'bg-red-50',
        border: 'border-red-200',
        icon: 'bg-red-100 text-red-600',
        title: 'text-red-800'
      }
    }
    return colors[color as keyof typeof colors] || colors.green
  }

  return (
    <div className="section bg-white">
      <div className="container max-w-6xl">
        <div className="text-center mb-12">
          <h1 className="text-4xl lg:text-5xl font-bold mb-4">
            Table of Contents
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            A comprehensive guide to agentic patterns organized into four progressive parts, 
            covering 21 essential chapters with practical code examples.
          </p>
        </div>

        <div className="space-y-8">
          {parts.map((part, partIndex) => {
            const IconComponent = part.icon
            const colorClasses = getColorClasses(part.color)
            
            return (
              <div key={part.id} className={`${colorClasses.bg} ${colorClasses.border} border rounded-xl p-8`}>
                <div className="flex items-start gap-4 mb-6">
                  <div className={`w-12 h-12 ${colorClasses.icon} rounded-lg flex items-center justify-center flex-shrink-0`}>
                    <IconComponent size={24} />
                  </div>
                  <div>
                    <h2 className={`text-2xl font-bold ${colorClasses.title} mb-2`}>
                      {part.title}
                    </h2>
                    <p className="text-gray-600">{part.description}</p>
                  </div>
                </div>

                <div className="grid md:grid-cols-2 gap-4">
                  {part.chapters.map((chapter) => (
                    <Link
                      key={chapter.id}
                      href={`/chapters/${chapter.id}`}
                      className="group bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-all duration-200 hover:border-gray-300"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <span className="w-8 h-8 bg-gray-100 text-gray-600 rounded-full flex items-center justify-center text-sm font-semibold">
                            {chapter.number}
                          </span>
                          <div>
                            <h3 className="font-semibold text-gray-900 group-hover:text-primary-600 transition-colors">
                              {chapter.title}
                            </h3>
                            {chapter.hasCode && (
                              <div className="flex items-center gap-1 mt-1">
                                <Code size={14} className="text-gray-500" />
                                <span className="text-xs text-gray-500">includes code examples</span>
                              </div>
                            )}
                          </div>
                        </div>
                        <ChevronRight size={16} className="text-gray-400 group-hover:text-primary-600 transition-colors" />
                      </div>
                    </Link>
                  ))}
                </div>
              </div>
            )
          })}
        </div>

        {/* Appendix Section */}
        <div className="mt-12 bg-gray-50 border border-gray-200 rounded-xl p-8">
          <div className="flex items-start gap-4 mb-6">
            <div className="w-12 h-12 bg-gray-100 text-gray-600 rounded-lg flex items-center justify-center">
              <FileText size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-800 mb-2">
                Appendix – Extended Topics and Practical Resources
              </h2>
              <p className="text-gray-600">
                Additional resources, advanced techniques, and practical implementation guides.
              </p>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            {appendices.map((appendix, index) => (
              <Link
                key={appendix.id}
                href={`/appendix/${appendix.id}`}
                className="group bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-all duration-200 hover:border-gray-300"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="w-8 h-8 bg-gray-100 text-gray-600 rounded-full flex items-center justify-center text-sm font-semibold">
                      {String.fromCharCode(65 + index)}
                    </span>
                    <div>
                      <h3 className="font-semibold text-gray-900 group-hover:text-primary-600 transition-colors">
                        {appendix.title}
                      </h3>
                      {appendix.onlineOnly && (
                        <span className="text-xs text-primary-600 bg-primary-50 px-2 py-1 rounded-full mt-1 inline-block">
                          online only
                        </span>
                      )}
                    </div>
                  </div>
                  <ChevronRight size={16} className="text-gray-400 group-hover:text-primary-600 transition-colors" />
                </div>
              </Link>
            ))}
          </div>
        </div>

        {/* Summary Stats */}
        <div className="mt-12 text-center">
          <div className="inline-flex items-center gap-8 bg-white rounded-xl border border-gray-200 px-8 py-6">
            <div>
              <div className="text-2xl font-bold text-primary-600">21</div>
              <div className="text-sm text-gray-600">Chapters</div>
            </div>
            <div className="w-px h-8 bg-gray-200"></div>
            <div>
              <div className="text-2xl font-bold text-primary-600">7</div>
              <div className="text-sm text-gray-600">Appendices</div>
            </div>
            <div className="w-px h-8 bg-gray-200"></div>
            <div>
              <div className="text-2xl font-bold text-primary-600">4</div>
              <div className="text-sm text-gray-600">Parts</div>
            </div>
            <div className="w-px h-8 bg-gray-200"></div>
            <div>
              <div className="text-2xl font-bold text-primary-600">21</div>
              <div className="text-sm text-gray-600">Code Examples</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
