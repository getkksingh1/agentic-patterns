import { ArrowLeft, ArrowRight, Code, Clock, Star } from 'lucide-react'
import Link from 'next/link'
import { Chapter } from '@/data/types'

interface ChapterContentProps {
  chapter: Chapter
}

export default function ChapterContent({ chapter }: ChapterContentProps) {
  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty.toLowerCase()) {
      case 'beginner':
        return 'bg-green-100 text-green-800'
      case 'intermediate':
        return 'bg-yellow-100 text-yellow-800'
      case 'advanced':
        return 'bg-red-100 text-red-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <div className="section bg-white">
      <div className="container max-w-4xl">
        {/* Chapter Header */}
        <div className="mb-8">
          <Link 
            href="/table-of-contents"
            className="inline-flex items-center text-primary-600 hover:text-primary-700 mb-4"
          >
            <ArrowLeft size={16} className="mr-2" />
            Back to Table of Contents
          </Link>
          
          <div className="flex items-center gap-4 mb-4">
            <span className="text-primary-600 font-medium">{chapter.part}</span>
            <span className="text-gray-300">•</span>
            <span className="text-gray-600">Chapter {chapter.number}</span>
          </div>
          
          <h1 className="text-4xl lg:text-5xl font-bold mb-4">
            {chapter.title}
          </h1>
          
          <p className="text-xl text-gray-600 mb-6">
            {chapter.description}
          </p>
          
          <div className="flex items-center gap-4 text-sm">
            <div className="flex items-center gap-2">
              <Clock size={16} className="text-gray-500" />
              <span className="text-gray-600">{chapter.readingTime}</span>
            </div>
            {chapter.difficulty && (
              <span className={`px-3 py-1 rounded-full text-xs ${getDifficultyColor(chapter.difficulty)}`}>
                {chapter.difficulty}
              </span>
            )}
          </div>
        </div>

        {/* Chapter Content */}
        <div className="prose prose-lg max-w-none">
          {/* Overview */}
          <section className="mb-8">
            <h2 className="text-2xl font-semibold mb-4">Overview</h2>
            <p className="text-gray-600 leading-relaxed">
              {chapter.overview || chapter.content?.overview}
            </p>
          </section>

          {/* Key Points */}
          <section className="mb-8">
            <h2 className="text-2xl font-semibold mb-4">Key Points</h2>
            <ul className="space-y-3">
              {(chapter.keyPoints || chapter.content?.keyPoints)?.map((point, index) => (
                <li key={index} className="flex items-start gap-3">
                  <Star className="text-primary-500 mt-1 flex-shrink-0" size={16} />
                  <span className="text-gray-600">{point}</span>
                </li>
              ))}
            </ul>
          </section>

          {/* Code Example */}
          {(chapter.codeExample || chapter.content?.codeExample) && (
            <section className="mb-8">
              <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
                <Code size={20} />
                Code Example
              </h2>
              <div className="bg-gray-900 rounded-lg p-6 overflow-x-auto">
                <pre className="text-green-400 text-sm">
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
