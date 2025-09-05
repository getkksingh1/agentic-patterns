import { ArrowLeft, ArrowRight, Code, Clock, Star, ChevronRight, Link as LinkIcon, Layers, Shuffle, Eye, Wrench, Target, Users, Brain, MessageSquare, Database, Zap, Shield, BarChart, CheckSquare, Search, Home, Book } from 'lucide-react'
import Link from 'next/link'
import { Chapter } from '@/data/types'

interface ChapterContentProps {
  chapter: Chapter
}

export default function ChapterContent({ chapter }: ChapterContentProps) {
  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty.toLowerCase()) {
      case 'beginner':
        return 'bg-green-100 text-green-800 border-green-200'
      case 'intermediate':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      case 'advanced':
        return 'bg-red-100 text-red-800 border-red-200'
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  const getPartTheme = (part: string) => {
    if (part?.includes('One')) {
      return {
        bg: 'bg-gradient-to-br from-green-50 to-emerald-50',
        border: 'border-green-200',
        accent: 'text-green-600',
        badgeBg: 'bg-green-100 text-green-800'
      }
    } else if (part?.includes('Two')) {
      return {
        bg: 'bg-gradient-to-br from-blue-50 to-cyan-50',
        border: 'border-blue-200',
        accent: 'text-blue-600',
        badgeBg: 'bg-blue-100 text-blue-800'
      }
    } else if (part?.includes('Three')) {
      return {
        bg: 'bg-gradient-to-br from-purple-50 to-indigo-50',
        border: 'border-purple-200',
        accent: 'text-purple-600',
        badgeBg: 'bg-purple-100 text-purple-800'
      }
    } else if (part?.includes('Four')) {
      return {
        bg: 'bg-gradient-to-br from-red-50 to-pink-50',
        border: 'border-red-200',
        accent: 'text-red-600',
        badgeBg: 'bg-red-100 text-red-800'
      }
    }
    return {
      bg: 'bg-gray-50',
      border: 'border-gray-200',
      accent: 'text-gray-600',
      badgeBg: 'bg-gray-100 text-gray-800'
    }
  }

  const getChapterIcon = (title: string, number: number) => {
    const iconMap: { [key: string]: any } = {
      'prompt chaining': LinkIcon,
      'routing': Shuffle,
      'parallelization': Layers,
      'reflection': Eye,
      'tool use': Wrench,
      'planning': Target,
      'multi-agent': Users,
      'memory management': Database,
      'learning and adaptation': Brain,
      'model context protocol': MessageSquare,
      'goal setting and monitoring': CheckSquare,
      'exception handling': Shield,
      'human-in-the-loop': Users,
      'knowledge retrieval': Search,
      'inter-agent communication': MessageSquare,
      'resource-aware optimization': Zap,
      'reasoning techniques': Brain,
      'guardrails': Shield,
      'evaluation and monitoring': BarChart,
      'prioritization': Target,
      'exploration and discovery': Search
    }
    
    const key = title.toLowerCase()
    const IconComponent = iconMap[key] || Book
    return IconComponent
  }

  const theme = getPartTheme(chapter.part || '')
  const ChapterIcon = getChapterIcon(chapter.title, chapter.number || 0)

  const getShortBenefitText = (title: string, description: string) => {
    const benefitMap: { [key: string]: string } = {
      'prompt chaining': 'Break down complex AI tasks into simple, reliable steps.',
      'routing': 'Intelligently direct queries to the right specialist every time.',
      'parallelization': 'Execute multiple AI tasks simultaneously for maximum efficiency.',
      'reflection': 'Enable AI systems to self-assess and continuously improve.',
      'tool use': 'Seamlessly integrate external APIs and functions into AI workflows.',
      'planning': 'Transform high-level goals into structured, executable action plans.',
      'multi-agent': 'Orchestrate teams of AI agents for collaborative problem-solving.'
    }
    
    return benefitMap[title.toLowerCase()] || description.split('.')[0] + '.'
  }

  return (
    <div className="min-h-screen bg-white">
      {/* Enhanced Breadcrumb Navigation */}
      <div className="bg-white border-b border-gray-200">
        <div className="container max-w-4xl py-4">
          <nav className="flex items-center text-sm text-gray-500 space-x-2">
            <Link href="/" className="hover:text-primary-600 transition-colors">
              <Home size={16} />
            </Link>
            <ChevronRight size={14} />
            <Link href="/table-of-contents" className="hover:text-primary-600 transition-colors">
              Guide
            </Link>
            <ChevronRight size={14} />
            <span className={theme.accent}>{chapter.part}</span>
            <ChevronRight size={14} />
            <span className="text-gray-900 font-medium">{chapter.title}</span>
          </nav>
        </div>
      </div>

      {/* Chapter Hero Section */}
      <div className={`${theme.bg} ${theme.border} border-b`}>
        <div className="container max-w-4xl py-12">
          {/* Navigation Buttons */}
          <div className="flex justify-between items-center mb-8">
            <Link 
              href="/table-of-contents"
              className="inline-flex items-center text-gray-600 hover:text-gray-900 transition-colors"
            >
              <ArrowLeft size={16} className="mr-2" />
              Table of Contents
            </Link>
            
            {/* Previous/Next Navigation */}
            <div className="flex items-center gap-4">
              {chapter.navigation?.previous && (
                <Link
                  href={(chapter.navigation.previous as any).href || `/chapters/${(chapter.navigation.previous as any).id}`}
                  className="inline-flex items-center px-4 py-2 bg-white rounded-lg border border-gray-200 hover:border-gray-300 transition-colors text-sm"
                >
                  <ArrowLeft size={14} className="mr-2" />
                  Previous
                </Link>
              )}
              {chapter.navigation?.next && (
                <Link
                  href={(chapter.navigation.next as any).href || `/chapters/${(chapter.navigation.next as any).id}`}
                  className="inline-flex items-center px-4 py-2 bg-white rounded-lg border border-gray-200 hover:border-gray-300 transition-colors text-sm"
                >
                  Next
                  <ArrowRight size={14} className="ml-2" />
                </Link>
              )}
            </div>
          </div>

          {/* Chapter Info */}
          <div className="flex items-start gap-6">
            {/* Chapter Icon */}
            <div className={`w-16 h-16 bg-white rounded-xl ${theme.border} border shadow-sm flex items-center justify-center flex-shrink-0`}>
              <ChapterIcon className={theme.accent} size={28} />
            </div>
            
            {/* Chapter Details */}
            <div className="flex-1">
              {/* Chapter Title */}
              <h1 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-3">
                {chapter.title}
              </h1>
              
              {/* Short Benefit-Driven Subtitle */}
              <p className="text-lg text-gray-700 mb-4 font-medium">
                {getShortBenefitText(chapter.title, chapter.description || '')}
              </p>
              
              {/* Metadata Badges */}
              <div className="flex items-center gap-3 mb-6">
                {chapter.difficulty && (
                  <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border ${getDifficultyColor(chapter.difficulty)}`}>
                    <Star size={14} className="mr-1" />
                    {chapter.difficulty}
                  </span>
                )}
                
                <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-100 text-gray-700 border border-gray-200">
                  <Clock size={14} className="mr-1" />
                  {chapter.readingTime}
                </span>
                
                <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border ${theme.badgeBg}`}>
                  <Book size={14} className="mr-1" />
                  {chapter.part}
                </span>
              </div>
              
              {/* "Why it matters" section */}
              <div className="bg-white/70 rounded-lg p-4 border border-white/50">
                <p className="text-base font-semibold text-gray-800 mb-2">
                  Why this pattern matters:
                </p>
                <p className="text-gray-700">
                  {chapter.description || 'This pattern is essential for building robust and scalable AI agent systems.'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container max-w-4xl py-8">
        {/* Key Highlights Box */}
        <div className="mb-8">
          <div className={`${theme.bg} rounded-xl p-6 ${theme.border} border`}>
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <Target className={theme.accent} size={20} />
              Key Use Cases & Applications
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium text-gray-800 mb-2">Perfect for:</h4>
                <ul className="text-sm text-gray-600 space-y-1">
                  {(chapter.practicalApplications || chapter.content?.practicalApplications || [
                    'Multi-step reasoning tasks',
                    'Complex workflow automation',
                    'Scalable AI agent systems'
                  ]).slice(0, 3).map((app, index) => (
                    <li key={index} className="flex items-start gap-2">
                      <span className={`${theme.accent} mt-1`}>•</span>
                      {app}
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <h4 className="font-medium text-gray-800 mb-2">You'll learn:</h4>
                <ul className="text-sm text-gray-600 space-y-1">
                  {(chapter.keyPoints || chapter.content?.keyPoints || [
                    'Core implementation patterns',
                    'Best practices and pitfalls',
                    'Real-world code examples'
                  ]).slice(0, 3).map((point, index) => (
                    <li key={index} className="flex items-start gap-2">
                      <span className={`${theme.accent} mt-1`}>•</span>
                      {point}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Chapter Content */}
        <div className="prose prose-lg max-w-none">
          {/* Overview */}
          <section className="mb-8">
            <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
              <Book className="text-primary-600" size={24} />
              Overview
            </h2>
            <div className="bg-gray-50 rounded-lg p-6 border border-gray-200">
              <p className="text-gray-700 leading-relaxed text-base">
                {chapter.overview || chapter.content?.overview}
              </p>
            </div>
          </section>

          {/* Key Points */}
          <section className="mb-8">
            <h2 className="text-2xl font-semibold mb-6 flex items-center gap-2">
              <Star className="text-yellow-500" size={24} />
              Essential Concepts
            </h2>
            <div className="grid gap-4">
              {(chapter.keyPoints || chapter.content?.keyPoints)?.map((point, index) => (
                <div key={index} className="bg-white rounded-lg p-4 border border-gray-200 shadow-sm">
                  <div className="flex items-start gap-3">
                    <div className={`w-6 h-6 ${theme.bg} ${theme.border} border rounded-full flex items-center justify-center flex-shrink-0 mt-0.5`}>
                      <span className={`text-xs font-bold ${theme.accent}`}>{index + 1}</span>
                    </div>
                    <p className="text-gray-700 leading-relaxed">{point}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Code Example */}
          {(chapter.codeExample || chapter.content?.codeExample) && (
            <section className="mb-8">
              <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
                <Code className="text-green-600" size={24} />
                Implementation Example
              </h2>
              <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl p-6 overflow-x-auto border border-gray-700 shadow-lg">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                    <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  </div>
                  <span className="text-gray-400 text-xs">Python</span>
                </div>
                <pre className="text-green-400 text-sm font-mono leading-relaxed">
                  <code>{chapter.codeExample || chapter.content?.codeExample}</code>
                </pre>
              </div>
            </section>
          )}

          {/* Practical Applications */}
          <section className="mb-8">
            <h2 className="text-2xl font-semibold mb-4">Practical Applications</h2>
            <div className="grid md:grid-cols-2 gap-4">
              {(chapter.practicalApplications || chapter.content?.practicalApplications)?.map((application, index) => (
                <div key={index} className="bg-primary-50 border border-primary-200 rounded-lg p-4">
                  <p className="text-gray-700">{application}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Additional Sections */}
          {chapter.sections && chapter.sections.map((section, index) => (
            <section key={index} className="mb-8">
              <h2 className="text-2xl font-semibold mb-4">{section.title}</h2>
              <div className="text-gray-600 leading-relaxed prose max-w-none">
                {section.content.split('\n\n').map((paragraph, pIndex) => {
                  // Handle markdown formatting
                  if (paragraph.includes('**') || paragraph.includes('```')) {
                    return (
                      <div key={pIndex} className="mb-4" dangerouslySetInnerHTML={{
                        __html: paragraph
                          .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                          .replace(/```json\n([\s\S]*?)\n```/g, '<pre class="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto"><code>$1</code></pre>')
                          .replace(/```([\s\S]*?)```/g, '<pre class="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto"><code>$1</code></pre>')
                          .replace(/`([^`]+)`/g, '<code class="bg-gray-100 px-2 py-1 rounded text-sm">$1</code>')
                      }} />
                    )
                  }
                  return <p key={pIndex} className="mb-4">{paragraph}</p>
                })}
              </div>
            </section>
          ))}

          {/* Practical Examples */}
          {chapter.practicalExamples && (
            <section className="mb-8">
              <h2 className="text-2xl font-semibold mb-4">Real-World Examples</h2>
              <div className="grid gap-6">
                {chapter.practicalExamples.map((example, index) => (
                  <div key={index} className="bg-gray-50 border border-gray-200 rounded-lg p-6">
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">{example.title}</h3>
                    <p className="text-gray-600 mb-4">{example.description}</p>
                    {(example.implementation || (example as any).example) && (
                      <div className="bg-white border border-gray-200 rounded p-3 mb-4 italic text-gray-700">
                        "{example.implementation || (example as any).example}"
                      </div>
                    )}
                    {(example as any).steps && (
                      <div className="space-y-2">
                        <h4 className="font-medium text-gray-900">Implementation Steps:</h4>
                        <ol className="space-y-1">
                          {((example as any).steps)?.map((step: string, stepIndex: number) => (
                            <li key={stepIndex} className="flex items-start gap-3">
                              <span className="w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-sm font-semibold mt-0.5 flex-shrink-0">
                                {stepIndex + 1}
                              </span>
                              <span className="text-gray-600">{step}</span>
                            </li>
                          ))}
                        </ol>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Next Steps */}
          <section className="mb-8">
            <h2 className="text-2xl font-semibold mb-4">Next Steps</h2>
            <ol className="space-y-2">
              {(chapter.nextSteps || chapter.content?.nextSteps)?.map((step, index) => (
                <li key={index} className="flex items-start gap-3">
                  <span className="w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-sm font-semibold mt-0.5">
                    {index + 1}
                  </span>
                  <span className="text-gray-600">{step}</span>
                </li>
              ))}
            </ol>
          </section>

          {/* References */}
          {chapter.references && (
            <section className="mb-8">
              <h2 className="text-2xl font-semibold mb-4">References</h2>
              <ul className="space-y-2">
                {chapter.references.map((reference, index) => {
                  // Parse reference text to extract title and URL
                  const urlMatch = reference.match(/(.*?):\s*(https?:\/\/[^\s]+)/);
                  if (urlMatch) {
                    const [, title, url] = urlMatch;
                    return (
                      <li key={index} className="flex items-start gap-3">
                        <span className="text-primary-500 mt-2 flex-shrink-0">•</span>
                        <div className="text-gray-600">
                          <span className="font-medium">{title.trim()}:</span>{' '}
                          <a 
                            href={url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="text-primary-600 hover:text-primary-700 underline break-all"
                          >
                            {url}
                          </a>
                        </div>
                      </li>
                    );
                  }
                  
                  // Fallback for references without clear title:url format
                  return (
                    <li key={index} className="flex items-start gap-3">
                      <span className="text-primary-500 mt-2 flex-shrink-0">•</span>
                      <span className="text-gray-600 break-all">{reference}</span>
                    </li>
                  );
                })}
              </ul>
            </section>
          )}

          {/* Summary Box */}
          <section className="mb-8">
            <div className="bg-primary-50 border border-primary-200 rounded-xl p-6">
              <h2 className="text-xl font-semibold text-primary-800 mb-3">📋 Chapter Summary</h2>
              <div className="text-primary-700 space-y-2">
                <p><strong>What:</strong> Prompt chaining breaks complex tasks into sequential, manageable steps for more reliable AI interactions.</p>
                <p><strong>Why:</strong> Single prompts often fail for complex tasks due to instruction neglect, contextual drift, and error propagation.</p>
                <p><strong>How:</strong> Create focused, sequential workflows where each step builds on the previous output.</p>
                <p><strong>When:</strong> Use for multi-step reasoning, tool integration, and building sophisticated agentic systems.</p>
              </div>
            </div>
          </section>
        </div>

        {/* Navigation */}
        <div className="flex justify-between items-center pt-8 mt-8 border-t border-gray-200">
          {chapter.navigation.previous ? (
            <Link
              href={(chapter.navigation.previous as any).href || `/chapters/${(chapter.navigation.previous as any).id}`}
              className="flex items-center gap-2 text-primary-600 hover:text-primary-700 font-medium"
            >
              <ArrowLeft size={16} />
              <div className="text-left">
                <div className="text-sm text-gray-500">Previous</div>
                <div>{chapter.navigation.previous.title}</div>
              </div>
            </Link>
          ) : (
            <div></div>
          )}

          {chapter.navigation.next ? (
            <Link
              href={(chapter.navigation.next as any).href || `/chapters/${(chapter.navigation.next as any).id}`}
              className="flex items-center gap-2 text-primary-600 hover:text-primary-700 font-medium text-right"
            >
              <div className="text-right">
                <div className="text-sm text-gray-500">Next</div>
                <div>{chapter.navigation.next.title}</div>
              </div>
              <ArrowRight size={16} />
            </Link>
          ) : (
            <div></div>
          )}
        </div>
      </div>
    </div>
  )
}
