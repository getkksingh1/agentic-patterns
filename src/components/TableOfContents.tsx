'use client'

import Link from 'next/link'
import { useState } from 'react'
import { Book, Code, Users, Shield, FileText, ChevronRight, ChevronDown, Star, Zap, Brain, Target } from 'lucide-react'

interface Chapter {
  id: string
  number: number
  title: string
  description: string
  hasCode?: boolean
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced'
  readingTime: string
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
      { id: 'prompt-chaining', number: 1, title: 'Prompt Chaining', description: 'Link prompts for multi-step reasoning workflows', hasCode: true, difficulty: 'Beginner', readingTime: '15 min' },
      { id: 'routing', number: 2, title: 'Routing', description: 'Intelligently direct queries to specialized handlers', hasCode: true, difficulty: 'Beginner', readingTime: '18 min' },
      { id: 'parallelization', number: 3, title: 'Parallelization', description: 'Execute multiple tasks simultaneously for efficiency', hasCode: true, difficulty: 'Intermediate', readingTime: '20 min' },
      { id: 'reflection', number: 4, title: 'Reflection', description: 'Self-assessment and iterative improvement mechanisms', hasCode: true, difficulty: 'Intermediate', readingTime: '22 min' },
      { id: 'tool-use', number: 5, title: 'Tool Use', description: 'Function calling and external API integration', hasCode: true, difficulty: 'Intermediate', readingTime: '25 min' },
      { id: 'planning', number: 6, title: 'Planning', description: 'Strategic goal decomposition and execution', hasCode: true, difficulty: 'Advanced', readingTime: '28 min' },
      { id: 'multi-agent', number: 7, title: 'Multi-Agent', description: 'Coordinated collaboration between multiple agents', hasCode: true, difficulty: 'Advanced', readingTime: '30 min' },
    ]
  },
  {
    id: 'learning',
    title: 'Part Two – Learning and Adaptation',
    description: 'Memory, learning, and adaptive behaviors',
    icon: Code,
    color: 'blue',
    chapters: [
      { id: 'memory-management', number: 8, title: 'Memory Management', description: 'Context persistence and information retrieval', hasCode: true, difficulty: 'Intermediate', readingTime: '24 min' },
      { id: 'learning-adaptation', number: 9, title: 'Learning and Adaptation', description: 'Continuous improvement from experience', hasCode: true, difficulty: 'Advanced', readingTime: '26 min' },
      { id: 'model-context-protocol', number: 10, title: 'Model Context Protocol (MCP)', description: 'Standardized context sharing across systems', hasCode: true, difficulty: 'Advanced', readingTime: '22 min' },
      { id: 'goal-setting-monitoring', number: 11, title: 'Goal Setting and Monitoring', description: 'SMART objectives and progress tracking', hasCode: true, difficulty: 'Intermediate', readingTime: '20 min' },
    ]
  },
  {
    id: 'human-centric',
    title: 'Part Three – Human-Centric Patterns',
    description: 'Human-AI collaboration and interaction patterns',
    icon: Users,
    color: 'purple',
    chapters: [
      { id: 'exception-handling', number: 12, title: 'Exception Handling and Recovery', description: 'Graceful failure management and recovery', hasCode: true, difficulty: 'Intermediate', readingTime: '18 min' },
      { id: 'human-in-loop', number: 13, title: 'Human-in-the-Loop', description: 'Seamless human-AI collaboration workflows', hasCode: true, difficulty: 'Beginner', readingTime: '16 min' },
      { id: 'knowledge-retrieval', number: 14, title: 'Knowledge Retrieval (RAG)', description: 'Advanced RAG and knowledge integration', hasCode: true, difficulty: 'Advanced', readingTime: '32 min' },
    ]
  },
  {
    id: 'scaling-safety',
    title: 'Part Four – Scaling, Safety, and Discovery',
    description: 'Enterprise-grade patterns for production systems',
    icon: Shield,
    color: 'red',
    chapters: [
      { id: 'inter-agent-communication', number: 15, title: 'Inter-Agent Communication (A2A)', description: 'Agent-to-agent messaging and protocols', hasCode: true, difficulty: 'Advanced', readingTime: '24 min' },
      { id: 'resource-aware-optimization', number: 16, title: 'Resource-Aware Optimization', description: 'Cost-effective resource management', hasCode: true, difficulty: 'Advanced', readingTime: '26 min' },
      { id: 'reasoning-techniques', number: 17, title: 'Reasoning Techniques', description: 'CoT, ReAct, and advanced reasoning patterns', hasCode: true, difficulty: 'Advanced', readingTime: '35 min' },
      { id: 'guardrails-safety', number: 18, title: 'Guardrails / Safety Patterns', description: 'Safety mechanisms and ethical constraints', hasCode: true, difficulty: 'Advanced', readingTime: '28 min' },
      { id: 'evaluation-monitoring', number: 19, title: 'Evaluation and Monitoring', description: 'Performance metrics and system monitoring', hasCode: true, difficulty: 'Advanced', readingTime: '30 min' },
      { id: 'prioritization', number: 20, title: 'Prioritization', description: 'Task scheduling and resource allocation', hasCode: true, difficulty: 'Intermediate', readingTime: '22 min' },
      { id: 'exploration-discovery', number: 21, title: 'Exploration and Discovery', description: 'Autonomous pattern discovery and innovation', hasCode: true, difficulty: 'Advanced', readingTime: '34 min' },
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
  const [expandedParts, setExpandedParts] = useState<string[]>(['foundations'])
  const [activeSection, setActiveSection] = useState<string>('foundations')

  const togglePart = (partId: string) => {
    setExpandedParts(prev =>
      prev.includes(partId)
        ? prev.filter(id => id !== partId)
        : [...prev, partId]
    )
  }

  const scrollToPart = (partId: string) => {
    setActiveSection(partId)
    const element = document.getElementById(partId)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  const getDifficultyBadge = (difficulty: 'Beginner' | 'Intermediate' | 'Advanced') => {
    const badges = {
      'Beginner': { color: 'bg-green-100 text-green-800', icon: <Star size={12} /> },
      'Intermediate': { color: 'bg-yellow-100 text-yellow-800', icon: <Zap size={12} /> },
      'Advanced': { color: 'bg-red-100 text-red-800', icon: <Brain size={12} /> }
    }
    return badges[difficulty]
  }

  const getColorClasses = (color: string) => {
    const colors = {
      green: {
        bg: 'bg-gradient-to-br from-green-50 to-emerald-50',
        border: 'border-green-200',
        icon: 'bg-gradient-to-br from-green-500 to-emerald-600 text-white',
        title: 'text-green-800',
        button: 'hover:bg-green-100'
      },
      blue: {
        bg: 'bg-gradient-to-br from-blue-50 to-cyan-50',
        border: 'border-blue-200',
        icon: 'bg-gradient-to-br from-blue-500 to-cyan-600 text-white',
        title: 'text-blue-800',
        button: 'hover:bg-blue-100'
      },
      purple: {
        bg: 'bg-gradient-to-br from-purple-50 to-indigo-50',
        border: 'border-purple-200',
        icon: 'bg-gradient-to-br from-purple-500 to-indigo-600 text-white',
        title: 'text-purple-800',
        button: 'hover:bg-purple-100'
      },
      red: {
        bg: 'bg-gradient-to-br from-red-50 to-pink-50',
        border: 'border-red-200',
        icon: 'bg-gradient-to-br from-red-500 to-pink-600 text-white',
        title: 'text-red-800',
        button: 'hover:bg-red-100'
      }
    }
    return colors[color as keyof typeof colors] || colors.green
  }

  return (
    <div className="section bg-white">
      {/* Sticky Navigation */}
      <div className="sticky top-0 z-50 bg-white/95 backdrop-blur-sm border-b border-gray-200 mb-8">
        <div className="container max-w-6xl">
          <div className="flex items-center justify-center gap-1 py-4 overflow-x-auto">
            {parts.map((part) => (
              <button
                key={part.id}
                onClick={() => scrollToPart(part.id)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 whitespace-nowrap ${
                  activeSection === part.id
                    ? 'bg-primary-100 text-primary-800'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                {part.title.split('–')[0].trim()}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="container max-w-6xl">
        <div className="text-center mb-12">
          <h1 className="text-4xl lg:text-5xl font-bold mb-4">
            Table of Contents
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            A comprehensive guide to agentic patterns organized into four progressive parts, 
            covering 21 essential chapters with practical code examples and implementations.
          </p>
        </div>

        <div className="space-y-8">
          {parts.map((part, partIndex) => {
            const IconComponent = part.icon
            const colorClasses = getColorClasses(part.color)
            const isExpanded = expandedParts.includes(part.id)
            
            return (
              <div key={part.id} id={part.id} className={`${colorClasses.bg} ${colorClasses.border} border rounded-xl overflow-hidden transition-all duration-300`}>
                {/* Part Header */}
                <button
                  onClick={() => togglePart(part.id)}
                  className={`w-full p-8 text-left ${colorClasses.button} transition-colors duration-200`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-start gap-4">
                      <div className={`w-16 h-16 ${colorClasses.icon} rounded-xl flex items-center justify-center flex-shrink-0 shadow-lg`}>
                        <IconComponent size={28} />
                      </div>
                      <div>
                        <h2 className={`text-2xl font-bold ${colorClasses.title} mb-2`}>
                          {part.title}
                        </h2>
                        <p className="text-gray-600 mb-3">{part.description}</p>
                        <div className="flex items-center gap-4 text-sm text-gray-500">
                          <span>{part.chapters.length} chapters</span>
                          <span>•</span>
                          <span>{part.chapters.reduce((acc, ch) => acc + parseInt(ch.readingTime), 0)} min total</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex-shrink-0 ml-4">
                      {isExpanded ? (
                        <ChevronDown size={24} className="text-gray-400" />
                      ) : (
                        <ChevronRight size={24} className="text-gray-400" />
                      )}
                    </div>
                  </div>
                </button>

                {/* Collapsible Chapter Content */}
                <div className={`overflow-hidden transition-all duration-300 ${
                  isExpanded ? 'max-h-none pb-8' : 'max-h-0'
                }`}>
                  <div className="px-8">
                    <div className="grid gap-4">
                      {part.chapters.map((chapter) => {
                        const difficultyBadge = getDifficultyBadge(chapter.difficulty)
                        return (
                          <Link
                            key={chapter.id}
                            href={`/chapters/${chapter.id}`}
                            className="group bg-white/80 backdrop-blur-sm border border-gray-200 rounded-lg p-5 hover:shadow-md transition-all duration-200 hover:border-gray-300 hover:bg-white"
                          >
                            <div className="flex items-start justify-between">
                              <div className="flex items-start gap-4 flex-1">
                                <span className="w-10 h-10 bg-gradient-to-br from-gray-100 to-gray-200 text-gray-700 rounded-lg flex items-center justify-center text-sm font-bold flex-shrink-0">
                                  {chapter.number}
                                </span>
                                <div className="flex-1">
                                  <h3 className="font-bold text-gray-900 group-hover:text-primary-600 transition-colors mb-1">
                                    {chapter.title}
                                  </h3>
                                  <p className="text-gray-600 text-sm mb-3">{chapter.description}</p>
                                  
                                  <div className="flex items-center gap-3 flex-wrap">
                                    <div className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${difficultyBadge.color}`}>
                                      {difficultyBadge.icon}
                                      {chapter.difficulty}
                                    </div>
                                    
                                    <div className="flex items-center gap-1 text-xs text-gray-500">
                                      <Target size={12} />
                                      {chapter.readingTime}
                                    </div>
                                    
                                    {chapter.hasCode && (
                                      <div className="flex items-center gap-1 text-xs text-primary-600 bg-primary-50 px-2 py-1 rounded-full">
                                        <Code size={12} />
                                        Code Examples
                                      </div>
                                    )}
                                  </div>
                                </div>
                              </div>
                              <ChevronRight size={16} className="text-gray-400 group-hover:text-primary-600 transition-colors flex-shrink-0 mt-1" />
                            </div>
                          </Link>
                        )
                      })}
                    </div>
                  </div>
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
